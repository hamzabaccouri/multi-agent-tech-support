import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from agents.AgentA import AgentA
from agents.AgentB import AgentB

# Load environment variables
load_dotenv()

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'agent_a' not in st.session_state:
        try:
            agent_b = AgentB(db_name="TechAssistDB")
            agent_b.initialize_knowledge_base("synthetic_data")
            st.session_state.agent_a = AgentA(agent_b=agent_b)
            st.session_state.agent_initialized = True
        except Exception as e:
            st.error(f"Error initializing agents: {str(e)}")
            st.session_state.agent_initialized = False

def reinitialize_agents():
    try:
        agent_b = AgentB(db_name="TechAssistDB")
        agent_b.initialize_knowledge_base("synthetic_data")
        st.session_state.agent_a = AgentA(agent_b=agent_b)
        st.session_state.agent_initialized = True
    except Exception as e:
        st.error(f"Error reinitializing agents: {str(e)}")
        st.session_state.agent_initialized = False

def save_chat_history():
    if st.session_state.messages:
        try:
            os.makedirs('chat_history', exist_ok=True)
            filename = f'chat_history/chat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                for msg in st.session_state.messages:
                    f.write(f"{msg['role']}: {msg['content']}\n")
            return True
        except Exception as e:
            st.error(f"Error saving chat history: {str(e)}")
            return False

def get_clean_response(response):
    if isinstance(response, dict):
        return response.get("answer", str(response))
    return str(response)

def load_selected_chat(selected_chat):
    try:
        with open(os.path.join('chat_history', selected_chat), 'r', encoding='utf-8') as f:
            chat_content = f.read()
            chat_messages = []
            for line in chat_content.strip().split('\n'):
                if line.startswith('user: '):
                    chat_messages.append({"role": "user", "content": line.replace('user: ', '')})
                elif line.startswith('assistant: '):
                    chat_messages.append({"role": "assistant", "content": line.replace('assistant: ', '')})
            st.session_state.messages = chat_messages
            st.success("Chat loaded successfully!")
    except Exception as e:
        st.error(f"Error loading chat: {str(e)}")

def main():
    st.set_page_config(
        page_title="Technical Support System",
        page_icon="💻",
        layout="wide"
    )

    initialize_session_state()

    with st.sidebar:
        st.title("Support Options")
        st.markdown("---")

        # Display the new chat icon, if found
        icon_path = "assets/new_chat_icon.png"
        if os.path.exists(icon_path):
            st.image(icon_path, width=24)
        else:
            st.warning("`new_chat_icon.png` not found in 'assets' directory.")

        if st.button("Start New Chat"):
            st.session_state.messages = []
            reinitialize_agents()
            st.success("Chat reset. Agents reinitialized!")

        st.markdown("---")

        if st.button("💾 Save Chat History"):
            if save_chat_history():
                st.success("Chat saved successfully!")

        st.markdown("---")

        if st.button("Load This Chat"):
            if os.path.exists('chat_history'):
                chat_files = sorted(os.listdir('chat_history'), reverse=True)
                if chat_files:
                    selected_chat = st.selectbox(
                        "Select a chat to load",
                        chat_files,
                        format_func=lambda x: x.replace('.txt', '').replace('chat_', '')
                    )
                    if selected_chat:
                        load_selected_chat(selected_chat)
                else:
                    st.info("No previous chats found.")
            else:
                st.info("No chat history directory found.")

        st.markdown("---")
        st.subheader("📚 Previous Chats")
        if os.path.exists('chat_history'):
            chat_files = sorted(os.listdir('chat_history'), reverse=True)
            if chat_files:
                selected_prev_chat = st.selectbox(
                    "Select a chat to view",
                    chat_files,
                    format_func=lambda x: x.replace('.txt', '').replace('chat_', '')
                )
                if selected_prev_chat:
                    with open(os.path.join('chat_history', selected_prev_chat), 'r', encoding='utf-8') as f:
                        chat_content = f.read()
                        st.text_area("Chat Content", chat_content, height=400, disabled=True)
            else:
                st.info("No previous chats found.")
        else:
            st.info("No chat history directory found.")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("MyApp Technical Support System\nVersion 1.0\n© 2024 MyApp")

    top_logo_col, top_title_col, _ = st.columns([1, 2, 1])
    with top_logo_col:
        logo_path = "assets/tech_support_logo.jpg"
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
        else:
            st.warning("Logo image not found. Please place your logo at 'assets/tech_support_logo.jpg'.")

    with top_title_col:
        st.title("MyApp Technical Support")

    st.markdown("---")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("How can I help you?"):
        if not st.session_state.get('agent_initialized', False):
            st.error("Support system is not properly initialized. Please try again later.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                try:
                    raw_response = st.session_state.agent_a.process_query(prompt)
                    clean_response = get_clean_response(raw_response)
                    st.session_state.messages.append({"role": "assistant", "content": clean_response})
                    st.write(clean_response)
                except Exception as e:
                    error_msg = "I apologize, but I encountered an error processing your request."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
