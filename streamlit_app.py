import streamlit as st
import requests
import datetime

# ============================================================================
# Configuration
# ============================================================================
API_URL = "http://localhost:8001/chat"

st.set_page_config(
    page_title="Agent Fraude Bancaire",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================================
# Sidebar - Context
# ============================================================================
st.sidebar.title("⚙️ Paramètres")

st.sidebar.subheader("Contexte Transaction")
with st.sidebar.form("context_form"):
    amount = st.number_input("Montant", value=149.99, step=0.01)
    currency = st.text_input("Devise", value="EUR")
    merchant = st.text_input("Marchand", value="AMAZON EU")
    channel = st.selectbox("Canal", options=["online", "in-store", "atm"], index=0)
    
    default_date = datetime.date(2026, 1, 20)
    trans_date = st.date_input("Date", value=default_date)
    
    country = st.text_input("Pays", value="LU")
    
    # Checkbox removed as per user request
    # fraud_confirmed = st.checkbox("Fraude Confirmée ?", value=True)
    
    update_context = st.form_submit_button("Appliquer le contexte")

if update_context:
    st.sidebar.success("Contexte mis à jour !")

# ============================================================================
# Main Chat Interface
# ============================================================================
st.title("🛡️ Agent Assistant Fraude")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial message
    st.session_state.messages.append({"role": "assistant", "content": "Merci pour ta confirmation de la fraude."})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Comment puis-je vous aider ?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare payload
    context_payload = {
        "amount": amount,
        "currency": currency,
        "merchant": merchant,
        "channel": channel,
        "date": str(trans_date),
        "country": country,
    }

    # History excluding the last message which is the current one
    # The API expects "conversation_history" to be the PREVIOUS turns
    history_payload = st.session_state.messages[:-1]

    payload = {
        "user_message": prompt,
        "fraud_confirmed": True, # Always True since user confirmed
        "transaction_context": context_payload,
        "conversation_history": history_payload,
    }

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
             with st.spinner("Analyse en cours..."):
                response = requests.post(API_URL, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                agent_response = data.get("agent_response", {})
                customer_message = agent_response.get("customer_message", "Désolé, je n'ai pas pu obtenir de réponse.")
                
                # Show main message
                message_placeholder.markdown(customer_message)
                
                # Show details (actions, citations) in expanders
                actions = agent_response.get("actions", [])
                if actions:
                    with st.expander("🛠️ Actions recommandées", expanded=True):
                        for action in actions:
                            st.write(f"- {action}")
                
                citations = agent_response.get("citations", [])
                if citations:
                    with st.expander("📚 Sources documentaires", expanded=False):
                        for cit in citations:
                            # Format nicely
                            filename = cit.get('filename', 'Inconnu')
                            page = cit.get('page_number', '?')
                            text = cit.get('text', '')
                            st.markdown(f"**{filename}** (p. {page})")
                            st.caption(f"> {text[:300]}...")
                            st.divider()

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": customer_message})

        except requests.exceptions.ConnectionError:
            message_placeholder.error("❌ Impossible de se connecter au serveur backend. Vérifiez qu'il est lancé sur http://localhost:8001")
        except Exception as e:
            message_placeholder.error(f"❌ Une erreur est survenue: {e}")
