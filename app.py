import streamlit as st
from streamlit_local_storage import LocalStorage

import os
import json
import time
import asyncio
import shutil
from dotenv import load_dotenv
from typing import List, Dict, AsyncGenerator
from datetime import datetime
from pathlib import Path
import tiktoken

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
BASE_HISTORY_DIR = Path("chat_histories")
BASE_HISTORY_DIR.mkdir(exist_ok=True) # 기본 대화 기록 저장 폴더 생성

global selected_category
global selected_item
selected_category = None
selected_item = None
llm_options = {
    "OpenAI":['gpt-5.1-2025-11-13','gpt-5-2025-08-07','gpt-4.1-nano','gpt-4.1-mini','gpt-4.1','gpt-4o','o4-mini','o3','o3-mini','o1','o1-mini'],
    "Gemini":['gemini-2.0-flash-001','gemini-2.5-flash','gemini-1.5-flash'],
    "Claude":['claude-3-7-sonnet-20250219', 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022','claude-3-5-sonnet-20240620','claude-sonnet-4-20250514']
}
#'claude-opus-4-20250514'

# --- 헬퍼 함수 ---
def get_user_history_dir() -> Path:
    """로그인된 사용자의 대화 기록 폴더 경로를 반환합니다."""
    if st.session_state.get("authenticated"):
        username = st.session_state.get("username", "default")
        user_dir = BASE_HISTORY_DIR / username
        user_dir.mkdir(exist_ok=True)
        return user_dir
    return BASE_HISTORY_DIR

def get_mcp_config_file() -> str:
    """로그인된 사용자의 mcp.json 파일 경로를 반환합니다."""
    if st.session_state.get("authenticated"):
        username = st.session_state.get("username", "default")
        return f"mcp_{username}.json"
    return "mcp.json"

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """주어진 텍스트의 토큰 수를 계산합니다."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def generate_filename_with_timestamp(prefix="chat_", extension="json"):
    """타임스탬프를 포함한 파일명을 생성합니다."""
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    if prefix:
        filename = f"{prefix}{timestamp_str}.{extension}"
    else:
        filename = f"{timestamp_str}.{extension}"
    return filename

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

def load_mcp_config():
    """사용자별 mcp.json 설정 파일을 로드하고 캐시합니다."""
    config_file = get_mcp_config_file()
    if not os.path.exists(config_file):
        # 사용자별 설정 파일이 없으면 기본 mcp.json으로 생성
        if os.path.exists("mcp.json"):
            shutil.copy("mcp.json", config_file)
            st.toast(f"'{config_file}'이(가) 없어 기본 설정으로 생성합니다.")
        else:
            # 기본 파일도 없으면 빈 설정으로 생성
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({"mcpServers": {}}, f, indent=2, ensure_ascii=False)
            st.toast(f"'{config_file}'이(가) 없어 빈 설정 파일로 생성합니다.")

    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_mcp_config(config):
    """MCP 서버 설정을 사용자별 mcp.json 파일에 저장합니다."""
    with open(get_mcp_config_file(), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def rename_chat(old_filename: str, new_filename_base: str):
    """대화 파일의 이름을 변경하고, 중복 시 숫자를 붙여 처리합니다."""
    HISTORY_DIR = get_user_history_dir()
    clean_base_name = new_filename_base.strip()
    if not clean_base_name:
        st.error("파일 이름은 비워둘 수 없습니다.")
        return

    new_filename = f"{clean_base_name}.json"
    old_path = HISTORY_DIR / old_filename
    new_path = HISTORY_DIR / new_filename

    if old_path == new_path: # 이름이 변경되지 않았으면 함수 종료
        return

    final_path = new_path
    final_filename = new_filename

    if final_path.exists():
        st.info(f"'{new_filename}' 파일이 이미 존재하여, 뒤에 숫자를 붙여 저장합니다.")
        counter = 1
        while True:
            unique_base_name = f"{clean_base_name} ({counter})"
            unique_filename = f"{unique_base_name}.json"
            unique_path = HISTORY_DIR / unique_filename
            if not unique_path.exists():
                final_path = unique_path
                final_filename = unique_filename
                break
            counter += 1

    try:
        old_path.rename(final_path)
        st.toast(f"'{old_filename}'을 '{final_filename}'(으)로 변경했습니다.")
        if st.session_state.get("current_chat_file") == old_filename:
            st.session_state.current_chat_file = final_filename
    except Exception as e:
        st.error(f"파일 이름 변경 중 오류 발생: {e}")

# --- 핵심 로직 함수 (기존과 동일하여 생략) ---
# --- 핵심 로직 함수 ---
async def plan_mcp_execution(query: str, servers_config: Dict) -> List[List[str]]:
    """사용자 질의와 도구 설명을 바탕으로 실행 계획(순차/병렬)을 수립합니다."""
    llm = get_llm()
    active_servers = {name: config for name, config in servers_config.items() if config.get("active", True)}

    if not active_servers:
        st.info("현재 활성화된 MCP 서버가 없습니다.")
        return []

    system_prompt = """You are an expert AI assistant that plans the execution flow for user requests using available tools.
    Analyze the user's query and the descriptions of available tools (MCP servers).
    Determine which tools are needed and the order of execution.

    Rules:
    1. If tasks depend on each other (e.g., Output of A is needed for B), schedule them sequentially.
    2. If tasks are independent (e.g., Compare A and B), schedule them in parallel (in the same step).
    3. Return the plan strictly as a JSON list of lists of server names.
       Example: [["server_A"], ["server_B", "server_C"], ["server_D"]]
       - Step 1: server_A runs.
       - Step 2: server_B and server_C run in parallel (after Step 1 finishes).
       - Step 3: server_D runs (after Step 2 finishes).
    4. If no tools are needed, return an empty list [].
    5. Only use the server names provided in the tool list. Do not invent new names.
    """
    
    prompt_template = """
    [Available Tools]
    {tools_description}

    [User Query]
    {user_query}

    [Execution Plan (JSON)]
    """
    
    descriptions = "\n".join([f"- {name}: {config['description']}" for name, config in active_servers.items()])
    prompt = ChatPromptTemplate.from_template(prompt_template).format(
        tools_description=descriptions,
        user_query=query
    )
    
    try:
        response = await llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        content = response.content.strip()
        # JSON 파싱 시도 (마크다운 코드 블록 제거)
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        plan = json.loads(content.strip())
        
        # 유효성 검사: 리스트의 리스트 형태인지 확인
        if isinstance(plan, list):
            validated_plan = []
            for step in plan:
                if isinstance(step, list):
                    # 실제 존재하는 서버만 필터링
                    valid_servers = [s for s in step if s in active_servers]
                    if valid_servers:
                        validated_plan.append(valid_servers)
                elif isinstance(step, str) and step in active_servers:
                     # 혹시 ["A", "B"] 처럼 1차원 리스트로 줬을 경우 대비 (모두 병렬로 처리하거나 순차로 처리? -> 여기선 단일 단계로 간주)
                     validated_plan.append([step])
            return validated_plan
        return []
    except Exception as e:
        st.error(f"실행 계획 수립 중 오류 발생: {e}")
        return []

# (★★★★★ 로직 수정 ★★★★★)
async def process_query(query: str, chat_history: List) -> AsyncGenerator[str, None]:
    """
    사용자 질의를 받아 서버 선택, 에이전트 생성 및 실행의 전체 과정을 처리합니다.
    'cancel scope' 오류를 해결하기 위해 단일 에이전트 실행 방식을 ainvoke로 변경합니다.
    """

    # <<< [수정] 대화 기록 관리 로직 시작 >>>
    MAX_HISTORY_TOKENS = 8192  # LLM에 전달할 최대 히스토리 토큰 수 제한

    history_for_llm = []
    current_tokens = 0

    # 전체 대화 기록을 최신순으로 순회하며 토큰 수를 확인
    for message in reversed(chat_history):
        message_content = message.content
        # 현재 메시지의 토큰 수를 계산
        message_tokens = count_tokens(message_content)

        # 이 메시지를 추가하면 최대 토큰 수를 넘는지 확인
        if current_tokens + message_tokens > MAX_HISTORY_TOKENS:
            # 넘는다면 더 이상 이전 기록을 추가하지 않고 종료
            break

        # 토큰 수 제한을 넘지 않으면 기록에 추가 (원본 순서를 위해 맨 앞에 삽입)
        history_for_llm.insert(0, message)
        current_tokens += message_tokens
    # <<< [수정] 대화 기록 관리 로직 끝 >>>

    mcp_config = load_mcp_config()["mcpServers"]
    llm = get_llm()

    # 1. 실행 계획 수립 (라우팅)
    st.write("`1. AI가 실행 계획을 수립 중입니다...`")
    execution_plan = await plan_mcp_execution(query, mcp_config)

    # 2. 연결할 MCP 서버가 없을 경우 (계획이 비어있음), LLM으로 직접 질의
    if not execution_plan:
        st.info("✅ LLM이 직접 답변합니다.")
        async for chunk in llm.astream(history_for_llm + [HumanMessage(content=query)]):
            yield chunk.content
        return

    # 3. 계획에 따른 단계별 실행
    st.write(f"`2. 수립된 계획: {execution_plan}`")
    
    accumulated_results = [] # 각 단계의 결과를 저장
    final_responses = {} # 최종 종합을 위한 응답 저장

    for step_idx, current_step_servers in enumerate(execution_plan):
        step_num = step_idx + 1
        st.write(f"`Step {step_num}: {', '.join(current_step_servers)} 실행 중...`")
        
        # 이전 단계까지의 결과 요약
        previous_context = ""
        if accumulated_results:
            previous_context = "\n\n[이전 단계 처리 결과]\n" + "\n".join(accumulated_results)

        async def run_agent_step(name: str, context: str) -> tuple[str, str]:
            """단일 에이전트 실행 (컨텍스트 포함)"""
            config = mcp_config[name]
            final_output = f"[{name}] 응답 없음"
            
            try:
                conn_type = config.get("transport")
                
                async def process_session(read, write):
                    nonlocal final_output
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await load_mcp_tools(session)
                        if not tools:
                            return f"[{name}] 도구 없음"
                        
                        agent = create_react_agent(llm, tools)
                        
                        # 에이전트에게 전달할 메시지 구성
                        # 이전 히스토리 + (이전 단계 결과가 포함된 시스템 메시지) + 현재 쿼리
                        system_msg = "당신은 사용자의 요청을 처리하는 에이전트입니다."
                        if context:
                            system_msg += f" 이전 단계에서 수행된 결과는 다음과 같습니다. 이를 바탕으로 작업을 수행하세요:\n{context}"
                        
                        step_messages = history_for_llm + [
                            SystemMessage(content=system_msg),
                            HumanMessage(content=query)
                        ]
                        
                        result = await agent.ainvoke({"messages": step_messages})
                        
                        if 'output' in result:
                            final_output = result['output']
                        elif 'messages' in result and isinstance(result['messages'][-1], AIMessage):
                            final_output = result['messages'][-1].content
                            
                if conn_type == "stdio":
                    params = StdioServerParameters(command=config.get("command"), args=config.get("args", []))
                    async with stdio_client(params) as (read, write):
                        await process_session(read, write)
                elif conn_type == "sse":
                    url = config.get("url")
                    headers = config.get("headers", {})
                    async with sse_client(url, headers=headers) as (read, write):
                        await process_session(read, write)
                        
            except Exception as e:
                final_output = f"[{name}] 실행 오류: {str(e)}"
                st.error(f"❌ '{name}' 실행 중 오류: {e}")
            
            return name, final_output

        # 현재 단계의 서버들 병렬 실행
        tasks = [run_agent_step(name, previous_context) for name in current_step_servers]
        results = await asyncio.gather(*tasks)
        
        # 결과 처리
        for name, output in results:
            # 결과 누적 (다음 단계를 위해)
            accumulated_results.append(f"Server '{name}' Output: {output}")
            
            # 최종 응답 딕셔너리에 저장 (마지막 종합을 위해)
            # 토큰 제한 처리
            MAX_RESPONSE_TOKENS = 1500
            if count_tokens(output) > MAX_RESPONSE_TOKENS:
                 final_responses[name] = output[:3000] + "...(생략)" # 대략적인 길이로 자름 (정확한 토큰 자르기는 생략하여 속도 향상)
            else:
                 final_responses[name] = output
            
            with st.expander(f"Step {step_num} - {name} 결과 확인"):
                st.write(output)

    # 4. 최종 답변 종합
    st.write("`3. 모든 단계 완료. 최종 답변 생성 중...`")
    
    history_str = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in chat_history])
    synthesis_prompt_template = """
    당신은 여러 AI 에이전트의 단계별 실행 결과를 종합하여 사용자에게 최종 답변을 제공하는 마스터 AI입니다.
    아래 대화 기록과 실행 계획에 따른 각 단계의 결과를 참고하여, 사용자의 원래 질문에 대한 완벽한 답변을 작성해주세요.
    
    [대화 기록]
    {chat_history}
    
    [사용자 질문]
    {original_query}
    
    [단계별 실행 결과]
    {agent_responses}
    
    [종합된 최종 답변]
    """
    synthesis_prompt = ChatPromptTemplate.from_template(synthesis_prompt_template)
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()
    
    # agent_responses를 보기 좋게 포맷팅
    formatted_responses = json.dumps(final_responses, ensure_ascii=False, indent=2)
    if accumulated_results:
         formatted_responses = "\n".join(accumulated_results)

    async for chunk in synthesis_chain.astream({
        "chat_history": history_str,
        "original_query": query,
        "agent_responses": formatted_responses
    }):
        yield chunk


# --- Streamlit UI 구성 ---
st.set_page_config(page_title="MCP Client on Streamlit", layout="wide")
st.title("🤖 MCP Client")

# --- 1. 인증 처리 (수정된 로직) ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

localS = LocalStorage()

if not st.session_state.authenticated:
    # 환경변수에서 사용자 정보 로드
    credentials_str = os.getenv("USER_CREDENTIALS", "")
    credentials = {}
    if credentials_str:
        for pair in credentials_str.split(','):
            try:
                username, password = pair.strip().split('|', 1)
                credentials[username] = password
            except ValueError:
                st.error("USER_CREDENTIALS 환경 변수 형식이 잘못되었습니다. 'id|pw,id2|pw2' 형식을 사용하세요.")
                st.stop()
    
    if not credentials:
        st.error("로그인 정보가 설정되지 않았습니다. USER_CREDENTIALS 환경 변수를 확인하세요.")
        st.stop()

    st.subheader("로그인")
    
    # localStorage에서 저장된 사용자 ID 불러오기
    remembered_username = localS.getItem("remembered_username") or ""
    
    # 저장된 아이디가 있으면 체크박스를 기본적으로 선택 상태로 둡니다.
    is_checked_by_default = remembered_username != ""
    
    username = st.text_input("사용자 아이디", value=remembered_username)
    remember_id = st.checkbox("아이디 저장", value=is_checked_by_default)
    password = st.text_input("비밀번호", type="password")
    
    if st.button("로그인"):
        if username in credentials and credentials[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            
            # '아이디 저장' 체크박스 상태에 따라 localStorage에 저장 또는 삭제
            if remember_id:
                localS.setItem("remembered_username", username)
            else:
                localS.setItem("remembered_username", "") # 저장된 아이디 삭제

            # (★★★ 수정된 부분 ★★★)
            # localStorage가 값을 설정할 수 있도록 아주 짧은 지연 시간을 추가합니다.
            time.sleep(0.1)
            
            st.rerun()
        else:
            st.error("아이디 또는 비밀번호가 일치하지 않습니다.")
    st.stop()

# --- 2. 메인 애플리케이션 (인증 후) ---
with st.sidebar:
    st.header(f"환영합니다, {st.session_state.username}님!")
    if st.button("로그아웃"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    def start_new_chat():
        st.session_state.messages = []
        st.session_state.current_chat_file = None

    def auto_save_chat():
        HISTORY_DIR = get_user_history_dir()
        if st.session_state.get("current_chat_file") and st.session_state.get("messages"):
            save_path = HISTORY_DIR / st.session_state.current_chat_file
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

    def load_chat(filename: str):
        HISTORY_DIR = get_user_history_dir()
        load_path = HISTORY_DIR / filename
        with open(load_path, "r", encoding="utf-8") as f:
            st.session_state.messages = json.load(f)
        st.session_state.current_chat_file = filename

    def delete_chat(filename: str):
        HISTORY_DIR = get_user_history_dir()
        if st.session_state.get("current_chat_file") == filename:
            start_new_chat()
        file_to_delete = HISTORY_DIR / filename
        if file_to_delete.exists():
            file_to_delete.unlink()
            st.toast(f"'{filename}'을 삭제했습니다.")

    st.button("새로운 채팅 열기", on_click=start_new_chat, use_container_width=True)
    st.divider()

    # LLM 관리 UI (기존과 동일)
    saved_model = localS.getItem("selected_model")
    saved_category = saved_model[0] if saved_model else ""
    saved_item = saved_model[1] if saved_model else ""
    
    categories = list(llm_options.keys())
    category_index = categories.index(saved_category) if saved_category in categories else 0
    
    st.header("LLM 관리")
    selected_category = st.selectbox("LLM를 선택하세요:", categories, index=category_index)
    
    model_options = llm_options[selected_category]
    item_index = model_options.index(saved_item) if saved_item in model_options else 0
    selected_item = st.selectbox(f"{selected_category} 중에서 선택하세요:", model_options, index=item_index)
    localS.setItem("selected_model", [selected_category,selected_item])

    st.divider()
    st.header(f"MCP 서버 관리 ({st.session_state.username})")
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

    # 대화 목록 관리 UI (기존과 동일, 경로만 수정됨)
    HISTORY_DIR = get_user_history_dir()
    if "editing_chat_file" not in st.session_state:
        st.session_state.editing_chat_file = None
    # ... (display_chat_item, show_all_chats_dialog 등 대화 목록 UI 함수는 기존과 동일) ...
    def display_chat_item(filename: str, key_prefix: str):
        """대화 목록 아이템을 표시하고 수정/삭제 UI를 제공하는 함수"""
        is_editing = st.session_state.get("editing_chat_file") == filename

        if is_editing:
            # 이름 수정 모드 UI
            c1, c2, c3 = st.columns([0.7, 0.15, 0.15])
            with c1:
                new_name_base = st.text_input(
                    "새 파일 이름",
                    value=filename.removesuffix(".json"),
                    key=f"text_{key_prefix}_{filename}",
                    label_visibility="collapsed"
                )
            with c2:
                if st.button("저장", key=f"save_{key_prefix}_{filename}", use_container_width=True, type="primary"):
                    rename_chat(filename, st.session_state[f"text_{key_prefix}_{filename}"])
                    st.session_state.editing_chat_file = None
                    st.rerun()
            with c3:
                if st.button("취소", key=f"cancel_{key_prefix}_{filename}", use_container_width=True):
                    st.session_state.editing_chat_file = None
                    st.rerun()
        else:
            # 일반 표시 모드 UI
            c1, c2, c3 = st.columns([0.75, 0.125, 0.125])
            with c1:
                is_active_chat = st.session_state.get("current_chat_file") == filename
                button_type = "primary" if is_active_chat else "secondary"
                if st.button(filename, key=f"load_{key_prefix}_{filename}", use_container_width=True, type=button_type):
                    if not is_active_chat:
                        load_chat(filename)
                        st.session_state.editing_chat_file = None
                        st.rerun()
            with c2:
                if st.button("✏️", key=f"edit_{key_prefix}_{filename}", use_container_width=True, help="이름 변경"):
                    st.session_state.editing_chat_file = filename
                    st.rerun()
            with c3:
                if st.button("X", key=f"delete_{key_prefix}_{filename}", use_container_width=True, help=f"{filename} 삭제"):
                    delete_chat(filename)
                    st.rerun()

    try:
        saved_chats_paths = [p for p in HISTORY_DIR.glob("*.json")]
        saved_chats_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        saved_chats = [p.name for p in saved_chats_paths]
    except FileNotFoundError:
        saved_chats = []

    @st.dialog("전체 대화 목록")
    def show_all_chats_dialog(older_chats_list):
        st.write(f"총 {len(saved_chats)}개의 대화가 있습니다.")
        items_to_show_count = st.session_state.get("dialog_items_to_show", 10)
        chats_to_display = older_chats_list[:items_to_show_count]
        for filename in chats_to_display:
            display_chat_item(filename, key_prefix="dialog")
        st.divider()
        if len(older_chats_list) > items_to_show_count:
            if st.button("더보기", use_container_width=True):
                st.session_state.dialog_items_to_show += 10
                st.rerun()
        if st.button("닫기", use_container_width=True, type="primary"):
            st.session_state.show_all_chats = False
            st.session_state.editing_chat_file = None
            st.rerun()
    
    if not saved_chats:
        st.write("저장된 대화가 없습니다.")
    else:
        recent_chats = saved_chats[:10]
        older_chats = saved_chats[10:]
        for filename in recent_chats:
            display_chat_item(filename, key_prefix="recent")
        if older_chats:
            if st.button("더 보기...", use_container_width=True):
                st.session_state.show_all_chats = True
                st.session_state.dialog_items_to_show = 10
                st.rerun()
    
    if st.session_state.get("show_all_chats"):
        older_chats = saved_chats[10:]
        show_all_chats_dialog(older_chats)


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
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # 핵심 로직 함수(process_query)가 생략되었으므로,
        # 원본 코드의 process_query 함수 전체를 위에 붙여넣어야 정상 동작합니다.
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        response = st.write_stream(process_query(prompt, history))
        st.badge("Answer by "+selected_item+"", icon=":material/check:", color="green")

    st.session_state.messages.append({"role": "assistant", "content": response})
    auto_save_chat()