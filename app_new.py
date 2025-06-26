import streamlit as st
import os
import json
import time
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any, AsyncGenerator
from datetime import datetime
from pathlib import Path

# LangChain 관련 라이브러리
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
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

# 전역 변수 대신 st.session_state를 사용하여 LLM 모델 선택을 관리합니다.
llm_options = {
    "OpenAI":['o4-mini','o3','o3-mini','o1','o1-mini','gpt-4o','gpt-4.1'],
    "Gemini":['gemini-2.0-flash-001','gemini-2.5-flash','gemini-1.5-flash'],
    "Claude":['claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022','claude-3-5-sonnet-20240620','claude-sonnet-4-20250514']
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

# @st.cache_resource 대신 st.session_state를 사용하여 LLM 인스턴스를 관리합니다.
def get_llm():
    """LLM 모델을 초기화합니다."""
    category = st.session_state.get('selected_llm_category', 'OpenAI')
    model_name = st.session_state.get('selected_llm_model', 'o4-mini')
    
    if category == 'Claude':
        llm = ChatAnthropic(model=model_name, temperature=0, max_tokens=4096)
    elif category == 'OpenAI':
        llm = ChatOpenAI(model=model_name, max_tokens=8000)
    elif category == 'Gemini':
        llm = ChatGoogleGenerativeAI(model=model_name)
    else: # 기본값
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

# --- ★★★★★ 핵심 로직 함수 (스트리밍 및 단일 에이전트 모델로 대폭 수정) ★★★★★ ---

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

async def process_query(query: str, chat_history: List) -> AsyncGenerator[str, None]:
    """
    사용자 질의를 받아 서버 선택, 에이전트 생성, 스트리밍 응답의 전체 과정을 처리합니다.
    """
    mcp_config = load_mcp_config()["mcpServers"]
    llm = get_llm()
    messages = chat_history + [HumanMessage(content=query)]

    # 1. MCP 서버 라우팅
    st.write("`1. MCP 서버 라우팅 중...`")
    selected_server_names = await select_mcp_servers(query, mcp_config)

    # 요청 1: 연결할 MCP 서버가 없을 경우, LLM으로 직접 질의 (스트리밍)
    if not selected_server_names:
        st.info("✅ 적합한 도구를 찾지 못했습니다. LLM이 직접 답변합니다.")
        async for chunk in llm.astream(messages):
            yield chunk.content
        return

    # 2. 선택된 서버에서 모든 도구를 수집
    st.write(f"`2. 선택된 서버: {', '.join(selected_server_names)}`")
    all_tools = []
    
    st.write("`3. 선택된 MCP 서버 세션 및 도구 로딩 중...`")
    
    # 비동기 작업을 병렬로 처리하여 더 빠른 로딩
    async def load_tools_from_server(name, config):
        try:
            conn_type = config.get("transport")
            if conn_type == "stdio":
                server_params = StdioServerParameters(command=config.get("command"), args=config.get("args", []))
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return await load_mcp_tools(session), name
            elif conn_type == "sse":
                url = config.get("url")
                headers = config.get("headers", {})
                async with sse_client(url, headers=headers) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return await load_mcp_tools(session), name
            else:
                st.warning(f"⚠️ '{name}' 서버의 연결 타입 ('{conn_type}')을 지원하지 않습니다.")
                return [], name
        except Exception as e:
            st.error(f"❌ '{name}' MCP 서버 연결 또는 도구 로딩 중 오류 발생: {e}")
            return [], name

    tasks = [load_tools_from_server(name, mcp_config[name]) for name in selected_server_names]
    results = await asyncio.gather(*tasks)

    for tools, name in results:
        if tools:
            all_tools.extend(tools)
            st.success(f"✅ '{name}' 서버 연결 및 도구 로드 성공: `{[tool.name for tool in tools]}`")
        else:
            st.warning(f"✅ '{name}' 서버에 연결했으나, 사용 가능한 도구가 없습니다.")
    
    # 3. 수집된 도구가 없으면 LLM으로 직접 답변 (스트리밍)
    if not all_tools:
        st.info("✅ 도구를 로드하지 못했습니다. LLM이 직접 답변합니다.")
        async for chunk in llm.astream(messages):
            yield chunk.content
        return

    # 4. 수집된 모든 도구를 사용하여 단일 에이전트 생성 및 실행
    st.write(f"`4. 총 {len(all_tools)}개 도구로 단일 에이전트 생성 및 질의 실행...`")
    
    try:
        agent = create_react_agent(llm, all_tools)
        agent_input = {"messages": messages}
        
        # ★★★★★ 해결 방안: astream_events를 사용하여 LLM의 최종 응답 토큰만 필터링 ★★★★★
        # version="v1"은 LangChain의 표준 이벤트 스키마를 사용하기 위함입니다.
        async for event in agent.astream_events(agent_input, version="v2"):
            
            kind = event["event"]
            
            # 에이전트 내의 Chat Model에서 토큰 스트림이 발생할 때만 처리합니다.
            # 이것이 바로 사용자가 보게 될 최종 응답의 실시간 스트림입니다.
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                # 비어있지 않은 토큰만 yield하여 클라이언트로 전송합니다.
                if content:
                    yield content

    except Exception as e:
        error_message = f"❌ 에이전트 실행 중 오류 발생: {e}"
        st.error(error_message)
        yield error_message


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
    
    selected_category = st.selectbox(
        "LLM를 선택하세요:", 
        list(llm_options.keys()), 
        key='selected_llm_category'
    )
    
    selected_item = st.selectbox(
        f"{selected_category} 중에서 선택하세요:", 
        llm_options[selected_category],
        key='selected_llm_model'
    )

    st.divider()
    st.header("MCP 서버 관리")
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

# --- ★★★★★ 사용자 입력 처리 (스트리밍 및 저장 로직으로 변경) ★★★★★ ---
if prompt := st.chat_input("질문을 입력하세요. (예: 서울 날씨 알려줘 그리고 거실 불 켜줘)"):
    if not st.session_state.get("current_chat_file"):
        st.session_state.current_chat_file = generate_filename_with_timestamp()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            history = [
                HumanMessage(content=m['content']) if m['role'] == 'user' else AIMessage(content=m['content'])
                for m in st.session_state.messages[:-1]
            ]
            
            # st.write_stream을 사용하여 실시간으로 응답을 표시하고, 최종 응답을 response 변수에 저장
            response_generator = process_query(prompt, history)
            response = st.write_stream(response_generator)

        st.badge(f"Answer by {st.session_state.get('selected_llm_model', 'N/A')}", icon=":material/check:", color="green")
    
    # 스트리밍이 완료된 후, 최종 응답을 대화 기록에 추가
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 대화 턴이 끝난 후, 자동 저장
    auto_save_chat()

    # st.rerun()을 호출하지 않아도 UI가 자연스럽게 업데이트됨