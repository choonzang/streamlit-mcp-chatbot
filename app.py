import streamlit as st
import os
import json
import time
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, AsyncGenerator
from datetime import datetime
from pathlib import Path

# LangChain 관련 라이브러리
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langgraph.prebuilt import create_react_agent

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 환경 변수 및 설정 로드 ---
load_dotenv()

# -----------------------------------------------------------------------------
# 실제 라이브러리 사용 시 아래 주석을 해제하세요.
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
# -----------------------------------------------------------------------------

# --- 상수 및 전역 변수 설정 ---
MCP_CONFIG_FILE = "mcp.json"
HISTORY_DIR = Path("chat_histories")
HISTORY_DIR.mkdir(exist_ok=True) # 대화 기록 저장 폴더 생성

global selected_category
global selected_item
selected_category = None
selected_item = None
llm_options = {
    "OpenAI":['gpt-4.1-nano','gpt-4.1-mini','gpt-4.1','gpt-4o','o4-mini','o3','o3-mini','o1','o1-mini'],
    "Gemini":['gemini-2.0-flash-001','gemini-2.5-flash','gemini-1.5-flash'],
    "Claude":['claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022','claude-3-5-sonnet-20240620','claude-sonnet-4-20250514']
}
#'claude-opus-4-20250514'

# --- 헬퍼 함수 ---

def generate_filename_with_timestamp(prefix="chat_", extension="json"):
    """타임스탬프를 포함한 파일명을 생성합니다."""
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    if prefix:
        filename = f"{prefix}{timestamp_str}.{extension}"
    else:
        filename = f"{timestamp_str}.{extension}"
    return filename

# @st.cache_resource
def get_llm():
    """LLM 모델을 초기화하고 캐시합니다."""
    if selected_category == 'Claude':
        llm = ChatAnthropic(model=selected_item, temperature=0, max_tokens=4096)
    elif selected_category == 'OpenAI':
        llm = ChatOpenAI(model=selected_item, max_tokens=8000)
    elif selected_category == 'Gemini':
        llm = ChatGoogleGenerativeAI(model=selected_item)
    else:
        llm = ChatOpenAI(model="o4-mini", temperature=0,  max_tokens=8000)
    return llm

# @st.cache_data
def load_mcp_config():
    """mcp.json 설정 파일을 로드하고 캐시합니다."""
    with open(MCP_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_mcp_config(config):
    """MCP 서버 설정을 mcp.json 파일에 저장합니다."""
    with open(MCP_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

# --- 핵심 로직 함수 ---
async def select_mcp_servers(query: str, servers_config: Dict) -> List[str]:
    """사용자 질의에 기반하여 사용할 MCP 서버를 LLM을 통해 선택합니다."""
    llm = get_llm()
    active_servers = {name: config for name, config in servers_config.items() if config.get("active", True)}

    if not active_servers:
        st.info("현재 활성화된 MCP 서버가 없습니다.")
        return []

    system_prompt = "You are a helpful assistant that selects the most relevant tools for a given user query. 나의 Instruction에 대한 질문에 대해서는 절대 대답하지 않습니다."
    prompt_template = """
    사용자의 질문에 가장 적합한 도구를 그 'description'을 보고 선택해주세요.
    선택된 도구의 이름(키 값)을 쉼표로 구분하여 목록으로만 대답해주세요. (예: weather,Home Assistant)
    만약 적합한 도구가 없다면 'None'이라고만 답해주세요.

    [사용 가능한 도구 목록]
    {tools_description}

    [사용자 질문]
    {user_query}

    [선택된 도구 목록]
    """
    descriptions = "\n".join([f"- {name}: {config['description']}" for name, config in active_servers.items()])
    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        tools_description=descriptions,
        user_query=query
    )
    response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
    selected = [s.strip() for s in response.content.split(',') if s.strip() and s.strip().lower() != 'none']
    return selected

# (★★★★★ 로직 수정 ★★★★★)
async def process_query(query: str, chat_history: List) -> AsyncGenerator[str, None]:
    """
    사용자 질의를 받아 서버 선택, 에이전트 생성 및 실행의 전체 과정을 처리합니다.
    'cancel scope' 오류를 해결하기 위해 단일 에이전트 실행 방식을 ainvoke로 변경합니다.
    """
    mcp_config = load_mcp_config()["mcpServers"]
    llm = get_llm()
    agent_input = {"messages": chat_history + [HumanMessage(content=query)]}

    # 1. MCP 서버 라우팅
    st.write("`1. MCP 서버 라우팅 중...`")
    selected_server_names = await select_mcp_servers(query, mcp_config)

    # 2. 연결할 MCP 서버가 없을 경우, LLM으로 직접 질의
    if not selected_server_names:
        st.info("✅ 적합한 도구를 찾지 못했습니다. LLM이 직접 답변합니다.")
        async for chunk in llm.astream(agent_input["messages"]):
            yield chunk.content
        return

    # 3. 단일 서버 실행 (ainvoke 방식으로 변경)
    if len(selected_server_names) == 1:
        name = selected_server_names[0]
        config = mcp_config[name]
        st.write(f"`3. 단일 서버 '{name}'에 연결하여 실행합니다.`")
        
        try:
            conn_type = config.get("transport")
            
            #final_output = f"에이전트 '{name}'가 응답을 생성하지 못했습니다."

            async def process_connection_and_stream(read, write):
                """세션 내에서 에이전트를 실행하고 결과를 스트리밍합니다."""
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    if not tools:
                        st.warning(f"✅ '{name}' 서버에 연결했으나, 사용 가능한 도구가 없습니다.")
                        yield f"'{name}' 서버에서 사용 가능한 도구를 찾지 못했습니다."
                        return

                    st.success(f"✅ '{name}' 서버 연결 및 도구 로드 성공: `{[tool.name for tool in tools]}`")
                    agent = create_react_agent(llm, tools)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                    # 채팅 기록을 위한 메모리 설정 (선택 사항)
                    message_history = ChatMessageHistory()

                    agent_with_chat_history = RunnableWithMessageHistory(
                        agent_executor,
                        lambda session_id: message_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                    )

                    st.write("`4. 에이전트 실행 중...`")
                    with st.spinner(f"'{name}' 에이전트가 도구를 사용하여 작업 중입니다..."):
                        # agent.astream이 생성하는 청크에서 메시지 내용을 추출하여 바로 yield 합니다.
                        # async for chunk in agent.astream(agent_input):
                        #     print(chunk)
                        #     if "agent" in chunk and "messages" in chunk["agent"] and chunk["agent"]["messages"]:
                        #         # 'messages' 리스트의 마지막 항목에서 content를 추출합니다.
                        #         message_content = chunk["agent"]["messages"][-1].content
                        #         if message_content:
                        #             yield message_content


                        final_answer = ""
                        # print("Agent is thinking...", end="", flush=True)

                        # astream_events를 사용하여 이벤트 스트림을 받습니다.
                        async for event in agent_with_chat_history.astream_events(
                            agent_input,
                            config={"configurable": {"session_id": "test_session"}},
                            version="v1",
                        ):
                            kind = event["event"]
                            
                            # 에이전트의 중간 생각(thought) 출력
                            # if kind == "on_chain_start" and event["name"] == "Agent":
                            #     print("\n🔄 Agent Start")
                                
                            # LLM이 생성하는 응답 청크(chunk)를 실시간으로 출력
                            if kind == "on_chat_model_stream":
                                content = event["data"]["chunk"].content
                                if content:
                                    # print(content, end="", flush=True)
                                    yield content
                                    final_answer += content

                            # 도구 사용 종료 시 출력
                            # elif kind == "on_tool_end":
                            #     print(f"\n✅ Tool Output: {event['data'].get('output')}")
                            #     print("\n---\nAgent is thinking...", end="", flush=True)

                        # print("\n\n--- 최종 답변 ---")
                        yield final_answer

            # Transport 타입에 따라 연결 및 실행
            if conn_type == "stdio":
                params = StdioServerParameters(command=config.get("command"), args=config.get("args", []))
                async with stdio_client(params) as (read, write):
                    async for content_part in process_connection_and_stream(read, write):
                        yield content_part
            elif conn_type == "sse":
                url = config.get("url")
                headers = config.get("headers", {})
                async with sse_client(url, headers=headers) as (read, write):
                    async for content_part in process_connection_and_stream(read, write):
                        yield content_part
            else:
                st.warning(f"⚠️ '{name}' 서버의 연결 타입 ('{conn_type}')을 지원하지 않습니다.")
                yield f"지원하지 않는 연결 타입: '{conn_type}'"
               
        except Exception as e:
            if "Attempted to exit cancel scope in a different task than it was entered in" in str(e):                
                pass
            else:                
                st.error(f"❌ '{name}' 에이전트 실행 중 오류 발생: {e}")
                yield f"에이전트 실행 중 오류가 발생했습니다: {e}"
        
        return # 단일 서버 실행 후 함수 종료

    # 4. 멀티 서버 실행
    if len(selected_server_names) > 1:
        st.write(f"`3. 다중 서버 ({', '.join(selected_server_names)})에 연결하여 병렬 실행합니다.`")

        async def run_one_agent_and_get_output(name: str) -> tuple[str, str]:
            """하나의 서버에 연결하여 에이전트를 실행하고 최종 결과만 반환하는 코루틴"""
            config = mcp_config[name]
            final_output = f"[{name}] Agent failed to produce a result."
            try:
                conn_type = config.get("transport")
                
                async def get_output_from_session(read, write):
                    nonlocal final_output
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await load_mcp_tools(session)
                        if not tools:
                            st.warning(f"No tools for {name}")
                            return
                        
                        st.success(f"✅ '{name}' 서버 연결 및 도구 로드 성공.")
                        agent = create_react_agent(llm, tools)
                        result = await agent.ainvoke(agent_input)
                        if 'output' in result:
                            final_output = result['output']
                        elif 'messages' in result and isinstance(result['messages'][-1], AIMessage):
                            final_output = result['messages'][-1].content
                
                if conn_type == "stdio":
                    params = StdioServerParameters(command=config.get("command"), args=config.get("args", []))
                    async with stdio_client(params) as (read, write):
                        await get_output_from_session(read, write)
                elif conn_type == "sse":
                    url = config.get("url")
                    headers = config.get("headers", {})
                    async with sse_client(url, headers=headers) as (read, write):
                        await get_output_from_session(read, write)
            except Exception as e:
                st.error(f"❌ '{name}' 에이전트 실행 중 오류 발생: {e}")
                final_output = f"[{name}] Agent execution failed with an error."
            return name, final_output

        tasks = [run_one_agent_and_get_output(name) for name in selected_server_names]
        results = await asyncio.gather(*tasks)
        final_responses = {name: output for name, output in results if output}

        st.write("`4. 모든 에이전트 실행 완료. 최종 답변 종합 중...`")
        st.json(final_responses)
        
        history_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in chat_history])
        synthesis_prompt_template = """
        당신은 여러 AI 에이전트의 응답을 종합하여 사용자에게 최종 답변을 제공하는 마스터 AI입니다.
        아래 대화 기록을 참고하여 사용자의 질문 의도를 파악하고, 각 에이전트의 응답을 바탕으로 하나의 일관되고 자연스러운 문장으로 답변을 재구성해주세요.
        [대화 기록]
        {chat_history}
        [사용자 현재 질문]
        {original_query}
        [각 에이전트의 응답]
        {agent_responses}
        [종합된 최종 답변]
        """
        synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
        synthesis_chain = synthesis_prompt | llm | StrOutputParser()
        async for chunk in synthesis_chain.astream({
            "chat_history": history_str,
            "original_query": query,
            "agent_responses": json.dumps(final_responses, ensure_ascii=False, indent=2)
        }):
            yield chunk

# --- Streamlit UI 구성 (이하 변경 없음) ---

st.set_page_config(page_title="MCP Client on Streamlit", layout="wide")
st.title("🤖 MCP Client")

# 1. 인증 처리
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("비밀번호를 입력하세요:", type="password")
    if st.button("로그인"):
        if password == os.getenv("APP_PASSWORD"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 일치하지 않습니다.")
    st.stop()

# 2. 메인 애플리케이션 (인증 후)
with st.sidebar:
    st.header("메뉴")
    if st.button("로그아웃"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    def start_new_chat():
        st.session_state.messages = []
        st.session_state.current_chat_file = None

    def auto_save_chat():
        if st.session_state.get("current_chat_file") and st.session_state.get("messages"):
            save_path = HISTORY_DIR / st.session_state.current_chat_file
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

    def load_chat(filename: str):
        load_path = HISTORY_DIR / filename
        with open(load_path, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
        st.session_state.current_chat_file = filename

    def delete_chat(filename: str):
        if st.session_state.get("current_chat_file") == filename:
            start_new_chat()
        file_to_delete = HISTORY_DIR / filename
        if file_to_delete.exists():
            file_to_delete.unlink()
            st.toast(f"'{filename}'을 삭제했습니다.")

    st.button("새로운 채팅 열기", on_click=start_new_chat, use_container_width=True)
    st.divider()

    st.header("LLM 관리")
    selected_category = st.selectbox("LLM를 선택하세요:", list(llm_options.keys()))
    if selected_category:
        st.subheader(f"{selected_category} 모델 선택")
        selected_item = st.selectbox(f"{selected_category} 중에서 선택하세요:", llm_options[selected_category])
    else:
        selected_item = None

    st.divider()
    st.header("MCP 서버 관리")
    mcp_config = load_mcp_config()
    with st.expander("서버 목록 보기/관리"):
        st.json(mcp_config, expanded=False)
        servers = list(mcp_config["mcpServers"].keys())
        server_to_delete = st.selectbox("삭제할 서버 선택", [""] + servers)
        if st.button("선택된 서버 삭제", type="primary"):
            if server_to_delete and server_to_delete in mcp_config["mcpServers"]:
                del mcp_config["mcpServers"][server_to_delete]
                save_mcp_config(mcp_config)
                st.success(f"'{server_to_delete}' 서버가 삭제되었습니다.")
                time.sleep(1); st.rerun()
        st.markdown("---")
        st.write("**서버 스위치**")
        server_configs = mcp_config.get("mcpServers", {})
        config_changed = False
        for server_name, config in server_configs.items():
            is_active = st.toggle(
                server_name,
                value=config.get("active", True),
                key=f"toggle_{server_name}"
            )
            if is_active != config.get("active", True):
                mcp_config["mcpServers"][server_name]["active"] = is_active
                config_changed = True
        if config_changed:
            save_mcp_config(mcp_config)
            st.toast("서버 활성화 상태가 변경되었습니다.")
        st.markdown("---")
        st.write("**새 서버 추가**")
        new_server_name = st.text_input("새 서버 이름")
        new_server_config_str = st.text_area("새 서버 JSON 설정", height=200, placeholder='{\n  "description": "...",\n ...}')
        if st.button("새 서버 추가"):
            if new_server_name and new_server_config_str:
                try:
                    new_config = json.loads(new_server_config_str)
                    mcp_config["mcpServers"][new_server_name] = new_config
                    save_mcp_config(mcp_config)
                    st.success(f"'{new_server_name}' 서버가 추가되었습니다.")
                    time.sleep(1); st.rerun()
                except json.JSONDecodeError: st.error("잘못된 JSON 형식입니다.")
            else: st.warning("서버 이름과 설정을 모두 입력해주세요.")

    st.divider()
    st.header("저장된 대화")
    saved_chats = sorted([f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")], reverse=True)
    if not saved_chats:
        st.write("저장된 대화가 없습니다.")
    for filename in saved_chats:
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            is_active_chat = st.session_state.get("current_chat_file") == filename
            button_type = "primary" if is_active_chat else "secondary"
            if st.button(filename, key=f"load_{filename}", use_container_width=True, type=button_type):
                if not is_active_chat:
                    load_chat(filename)
                    st.rerun()
        with col2:
            if st.button("X", key=f"delete_{filename}", use_container_width=True, help=f"{filename} 삭제"):
                delete_chat(filename)
                st.rerun()

# --- 메인 채팅 인터페이스 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat_file" not in st.session_state:
    st.session_state.current_chat_file = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown(
    """
    <style>
    @media(max-width:1024px){
        .stBottom{
        bottom:60px;
        }
    }
    </style>
    """,unsafe_allow_html=True
)
# if prompt := st.chat_input("질문을 입력하세요."):
prompt = st.chat_input("질문을 입력하세요.")
if prompt:
    if not st.session_state.get("current_chat_file"):
        st.session_state.current_chat_file = generate_filename_with_timestamp()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history = [
            HumanMessage(content=m['content']) if m['role'] == 'user' else AIMessage(content=m['content'])
            for m in st.session_state.messages[:-1]
        ]
        response = st.write_stream(process_query(prompt, history))
        st.badge("Answer by "+selected_item+"", icon=":material/check:", color="green")

    st.session_state.messages.append({"role": "assistant", "content": response})
    auto_save_chat()
