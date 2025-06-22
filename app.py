import streamlit as st
import os
import json
import time
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# LangChain 관련 라이브러리
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langgraph.prebuilt import create_react_agent

# --- 환경 변수 및 설정 로드 ---
load_dotenv()

# -----------------------------------------------------------------------------
# 실제 라이브러리 사용 시 아래 주석을 해제하고, 위의 Mock 객체들은 삭제하세요.
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
    "OpenAI":['o4-mini','o3','o3-mini','o1','o1-mini','gpt-4o','gpt-4.1'],
    "Claude":['claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022','claude-3-5-sonnet-20240620','claude-sonnet-4-20250514','claude-opus-4-20250514']
}

# --- 헬퍼 함수 ---

def run_async(func):
    """비동기 함수를 Streamlit에서 실행하기 위한 헬퍼"""
    return asyncio.run(func)

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
    else:
        llm = ChatOpenAI(model="o4-mini", max_tokens=8000)
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

# --- 핵심 로직 함수 (변경 없음) ---
# ... (기존 process_query, select_mcp_servers 함수들은 변경 없이 그대로 유지) ...
async def select_mcp_servers(query: str, servers_config: Dict) -> List[str]:
    """사용자 질의에 기반하여 사용할 MCP 서버를 LLM을 통해 선택합니다."""
    llm = get_llm()
    
    system_prompt = "You are a helpful assistant that selects the most relevant tools for a given user query."
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
    
    descriptions = "\n".join([f"- {name}: {config['description']}" for name, config in servers_config.items()])
    
    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        tools_description=descriptions,
        user_query=query
    )
    
    response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
    selected = [s.strip() for s in response.content.split(',') if s.strip() and s.strip().lower() != 'none']
    return selected

async def process_query(query: str, chat_history: List):
    """
    사용자 질의를 받아 서버 선택, 에이전트 생성, 질의 실행의 전체 과정을 처리합니다.
    """
    mcp_config = load_mcp_config()["mcpServers"]
    llm = get_llm()

    # 1. MCP 서버 라우팅
    st.write("`1. MCP 서버 라우팅 중...`")
    selected_server_names = await select_mcp_servers(query, mcp_config)

    # 요청 1: 연결할 MCP 서버가 없을 경우, LLM으로 직접 질의
    if not selected_server_names:
        st.info("✅ 적합한 도구를 찾지 못했습니다. LLM이 직접 답변합니다.")
        messages = chat_history + [HumanMessage(content=query)]
        response = await llm.ainvoke(messages)
        return response.content

    # 2. 선택된 서버 처리
    st.write(f"`2. 선택된 서버: {', '.join(selected_server_names)}`")
    agents = {}
    
    st.write("`3. 선택된 MCP 서버 세션 및 도구 로딩 중...`")

    for name in selected_server_names:
        config = mcp_config[name]
        tools = []

        try:
            conn_type = config.get("transport")

            if conn_type == "stdio":
                server_params = StdioServerParameters(command=config.get("command"), args=config.get("args", []))
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await load_mcp_tools(session)
                        if not tools:
                            st.warning(f"✅ '{name}' 서버에 연결했으나, 사용 가능한 도구가 없습니다.")
                            continue
                        st.success(f"✅ '{name}' 서버 연결 및 도구 로드 성공: `{[tool.name for tool in tools]}`")
                        agent = create_react_agent(llm, tools)
                        agent_input = {"messages": chat_history + [HumanMessage(content=query)]}
                        if len(selected_server_names) == 1:
                            response = await agent.ainvoke(agent_input)
                            return response.get('output', response['messages'][-1].content if 'messages' in response and isinstance(response['messages'][-1], AIMessage) else "응답 내용 파싱 실패")
                        else:
                            agents[name] = agent
            elif conn_type == "sse":
                url = config.get("url")
                headers = config.get("headers", {})
                async with sse_client(url, headers=headers) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await load_mcp_tools(session)
                        if not tools:
                            st.warning(f"✅ '{name}' 서버에 연결했으나, 사용 가능한 도구가 없습니다.")
                            continue
                        st.success(f"✅ '{name}' 서버 연결 및 도구 로드 성공: `{[tool.name for tool in tools]}`")
                        agent = create_react_agent(llm, tools)
                        agent_input = {"messages": chat_history + [HumanMessage(content=query)]}
                        if len(selected_server_names) == 1:
                            response = await agent.ainvoke(agent_input)
                            return response.get('output', response['messages'][-1].content if 'messages' in response and isinstance(response['messages'][-1], AIMessage) else "응답 내용 파싱 실패")
                        else:
                            agents[name] = agent
            else:
                st.warning(f"⚠️ '{name}' 서버의 연결 타입 ('{conn_type}')을 지원하지 않습니다.")
                continue
        except Exception as e:
            st.error(f"❌ '{name}' MCP 서버 연결 또는 도구 로딩 중 오류 발생: {e}")
            continue
    
    if not agents:
        return "선택된 모든 서버에 연결하지 못했거나, 에이전트를 생성할 수 있는 서버가 없습니다."

    st.write(f"`4. {len(agents)}개 에이전트 생성 완료. 질의 실행...`")
    parallel_runnable = RunnableParallel(**{name: agent for name, agent in agents.items()})
    
    try:
        all_agent_results = await parallel_runnable.ainvoke(agent_input)
        final_responses = {}
        for name, result in all_agent_results.items():
            if 'output' in result:
                final_responses[name] = result['output']
            elif 'messages' in result and isinstance(result['messages'][-1], AIMessage):
                final_responses[name] = result['messages'][-1].content
            else:
                final_responses[name] = f"[{name}] 응답 내용을 파싱할 수 없습니다."

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
        final_answer = await synthesis_chain.ainvoke({
            "chat_history": history_str,
            "original_query": query,
            "agent_responses": json.dumps(final_responses, ensure_ascii=False, indent=2)
        })
        return final_answer
    except Exception as e:
        st.error(f"❌ 병렬 에이전트 실행 중 오류 발생: {e}")
        return f"병렬 에이전트 실행 중 오류가 발생했습니다: {e}"

# --- Streamlit UI 구성 ---

st.set_page_config(page_title="MCP Client on Streamlit", layout="wide")
st.title("🤖 MCP(Model Context Protocol) 클라이언트")

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
# 사이드바 구성
with st.sidebar:
    st.header("메뉴")
    if st.button("로그아웃"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # --- 채팅 기록 관리 함수 (★★★★★ 로직 변경 ★★★★★) ---
    def start_new_chat():
        """세션을 초기화하여 새로운 채팅을 시작합니다."""
        st.session_state.messages = []
        st.session_state.current_chat_file = None

    def auto_save_chat():
        """현재 대화를 활성 파일에 자동으로 저장합니다."""
        if st.session_state.get("current_chat_file") and st.session_state.get("messages"):
            save_path = HISTORY_DIR / st.session_state.current_chat_file
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

    def load_chat(filename: str):
        """선택한 파일을 읽고 활성 채팅으로 설정합니다."""
        load_path = HISTORY_DIR / filename
        with open(load_path, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
        st.session_state.current_chat_file = filename

    def delete_chat(filename: str):
        """채팅 기록 파일을 삭제하고, 활성 세션이었다면 초기화합니다."""
        if st.session_state.get("current_chat_file") == filename:
            start_new_chat() # 활성 채팅을 삭제하면 화면도 초기화
        
        file_to_delete = HISTORY_DIR / filename
        if file_to_delete.exists():
            file_to_delete.unlink()
            st.toast(f"'{filename}'을 삭제했습니다.")

    st.button("새로운 채팅 열기", on_click=start_new_chat, use_container_width=True)
    st.divider()

    # --- LLM 및 MCP 서버 관리 ---
    st.header("LLM 관리")
    selected_category = st.selectbox("LLM를 선택하세요:", list(llm_options.keys()))
    if selected_category:
        st.subheader(f"{selected_category} 모델 선택")
        selected_item = st.selectbox(f"{selected_category} 중에서 선택하세요:", llm_options[selected_category])
    else:
        selected_item = None
    
    st.divider()
    st.header("MCP 서버 관리")
    # ... (기존 MCP 서버 관리 코드는 변경 없음) ...
    mcp_config = load_mcp_config()
    with st.expander("서버 목록 보기/관리"):
        st.json(mcp_config)
        servers = list(mcp_config["mcpServers"].keys())
        server_to_delete = st.selectbox("삭제할 서버 선택", [""] + servers)
        if st.button("선택된 서버 삭제", type="primary"):
            if server_to_delete and server_to_delete in mcp_config["mcpServers"]:
                del mcp_config["mcpServers"][server_to_delete]
                save_mcp_config(mcp_config)
                st.success(f"'{server_to_delete}' 서버가 삭제되었습니다.")
                time.sleep(1); st.rerun()
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

    # --- 저장된 대화 목록 표시 ---
    st.header("저장된 대화")
    saved_chats = sorted([f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")], reverse=True)
    
    if not saved_chats:
        st.write("저장된 대화가 없습니다.")
    
    for filename in saved_chats:
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            # 현재 활성 채팅은 다르게 표시 (예: 볼드체)
            is_active_chat = st.session_state.get("current_chat_file") == filename
            button_label = f"**{filename}**" if is_active_chat else filename
            if st.button(button_label, key=f"load_{filename}", use_container_width=True):
                load_chat(filename)
                st.rerun()
        with col2:
            if st.button("X", key=f"delete_{filename}", use_container_width=True, help=f"{filename} 삭제"):
                delete_chat(filename)
                st.rerun()


# --- 메인 채팅 인터페이스 ---

# 채팅 기록 및 활성 파일 초기화 (★★★★★ 로직 변경 ★★★★★)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat_file" not in st.session_state:
    st.session_state.current_chat_file = None

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리 (★★★★★ 로직 변경 ★★★★★)
if prompt := st.chat_input("질문을 입력하세요. (예: 서울 날씨 알려줘 그리고 거실 불 켜줘)"):
    # 1. 새 채팅인 경우, 활성 파일 이름 생성
    if not st.session_state.get("current_chat_file"):
        st.session_state.current_chat_file = generate_filename_with_timestamp()

    # 2. 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. 어시스턴트 응답 처리 및 표시
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            history = [
                HumanMessage(content=m['content']) if m['role'] == 'user' else AIMessage(content=m['content'])
                for m in st.session_state.messages[:-1]
            ]
            response = run_async(process_query(prompt, history))
            st.markdown(response)
        st.badge("Answer by "+selected_item+"", icon=":material/check:", color="green")
    
    # 4. 어시스턴트 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 5. 대화 턴이 끝난 후, 전체 대화를 활성 파일에 자동 저장
    auto_save_chat()

    # 6. 화면 새로고침 (저장된 파일 목록 업데이트 등)
    #st.rerun()