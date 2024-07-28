import streamlit as st
import json
from typing import Iterable
from moa.agent import MOAgent
from moa.agent.moa import ResponseChunk
from streamlit_ace import st_ace
import copy
from pydub import AudioSegment
import soundfile as sf
import sounddevice as sd
from groq import Groq
import os

# Default configuration
default_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 3,
    "layer_agent_config": {}
}

layer_agent_config_def = {
    
    "layer_agent_1": {
        "system_prompt": "Pense na sua resposta passo a passo. {helper_response}",
        "model_name": "llama-3.1-8b-instant"
    },
    "layer_agent_2": {
        "system_prompt": "Responda com um pensamento e, em seguida, sua resposta a pergunta. {helper_response}",
        "model_name": "gemma-7b-it",
        "temperature": 0.7
    },
    "layer_agent_3": {
        "system_prompt": "Voce e um especialista em logica e raciocinio. Sempre tome uma abordagem logica para a resposta. {helper_response}",
         "model_name": "llama3-8b-8192"
    }
}

# Recommended Configuration

rec_config = {
    "main_model": "llama3-70b-8192",
    "cycles": 2,
    "layer_agent_config": {}
}

layer_agent_config_rec = {
    "layer_agent_1": {
        "system_prompt": "Pense na sua resposta passo a passo. {helper_response}",
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0.1
    },
    "layer_agent_2": {
        "system_prompt": "Responda com um pensamento e, em seguida, sua resposta à pergunta. {helper_response}",
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0.2
    },
    "layer_agent_3": {
        "system_prompt": "Voce e um especialista em logica e raciocinio. Sempre tome uma abordagem logica para a resposta. {helper_response}",
        "model_name": "llama-3.1-8b-instant",
        "temperature": 0.4
    },
    "layer_agent_4": {
        "system_prompt": "Voce e um agente planejador especialista. Crie um plano de como responder a consulta do humano. {helper_response}",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.5
    }
}


def stream_response(messages: Iterable[ResponseChunk]):
    layer_outputs = {}
    for message in messages:
        if message['response_type'] == 'intermediate':
            layer = message['metadata']['layer']
            if layer not in layer_outputs:
                layer_outputs[layer] = []
            layer_outputs[layer].append(message['delta'])
        else:
            # Display accumulated layer outputs
            for layer, outputs in layer_outputs.items():
                st.write(f"Layer {layer}")
                cols = st.columns(len(outputs))
                for i, output in enumerate(outputs):
                    with cols[i]:
                        st.expander(label=f"Agent {i+1}", expanded=False).write(output)
            
            # Clear layer outputs for the next iteration
            layer_outputs = {}
            
            # Yield the main agent's output
            yield message['delta']

def set_moa_agent(
    main_model: str = default_config['main_model'],
    cycles: int = default_config['cycles'],
    layer_agent_config: dict[dict[str, any]] = copy.deepcopy(layer_agent_config_def),
    main_model_temperature: float = 0.1,
    override: bool = False
):
    if override or ("main_model" not in st.session_state):
        st.session_state.main_model = main_model
    else:
        if "main_model" not in st.session_state: st.session_state.main_model = main_model 

    if override or ("cycles" not in st.session_state):
        st.session_state.cycles = cycles
    else:
        if "cycles" not in st.session_state: st.session_state.cycles = cycles

    if override or ("layer_agent_config" not in st.session_state):
        st.session_state.layer_agent_config = layer_agent_config
    else:
        if "layer_agent_config" not in st.session_state: st.session_state.layer_agent_config = layer_agent_config

    if override or ("main_temp" not in st.session_state):
        st.session_state.main_temp = main_model_temperature
    else:
        if "main_temp" not in st.session_state: st.session_state.main_temp = main_model_temperature

    cls_ly_conf = copy.deepcopy(st.session_state.layer_agent_config)
    
    if override or ("moa_agent" not in st.session_state):
        st.session_state.moa_agent = MOAgent.from_config(
            main_model=st.session_state.main_model,
            cycles=st.session_state.cycles,
            layer_agent_config=cls_ly_conf,
            temperature=st.session_state.main_temp
        )

    del cls_ly_conf
    del layer_agent_config

st.set_page_config(
    page_title="Mistura de Agentes Powered by Groq",
    page_icon='static/favicon.ico',
        menu_items={
        'About': "## Groq Mistura de Agentes \n Powered by [Groq](https://groq.com)"
    },
    layout="wide"
)
valid_model_names = [
    'llama-3.1-70b-versatile',
    'llama-3.1-8b-instant',
    'llama3-70b-8192',
    'llama3-8b-8192',
    'gemma-7b-it',
    'gemma2-9b-it',
    'mixtral-8x7b-32768'
]

st.markdown("<a href='https://groq.com'><img src='app/static/banner.png' width='500'></a>", unsafe_allow_html=True)
st.write("---")



# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

set_moa_agent()

# Sidebar for configuration
with st.sidebar:
    # config_form = st.form("Agent Configuration", border=False)
    st.title("Configuração MOA")
    with st.form("Configuração de agentes", border=False):
        if st.form_submit_button("Utilizar Configuração recomendada"):
            try:
                set_moa_agent(
                    main_model=rec_config['main_model'],
                    cycles=rec_config['cycles'],
                    layer_agent_config=layer_agent_config_rec,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuração atualizada com sucesso!")
            except json.JSONDecodeError:
                st.error("JSON inválido na configuração do agente de camada. Por favor, verifique sua entrada.")
            except Exception as e:
                st.error(f"Erro atualizando configuração: {str(e)}")
        # Main model selection
        new_main_model = st.selectbox(
            "Selecionar Modelo principal",
            options=valid_model_names,
            index=valid_model_names.index(st.session_state.main_model)
        )

        # Cycles input
        new_cycles = st.number_input(
            "Numero de camadas",
            min_value=1,
            max_value=10,
            value=st.session_state.cycles
        )

        # Main Model Temperature
        main_temperature = st.number_input(
            label="Temperatura do modelo principal",
            value=0.1,
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        # Layer agent configuration
        tooltip = "Os agentes na configuração do agente de camada executam em paralelo por ciclo. Cada agente de camada suporta todos os parâmetros de inicialização da classe [Langchain's ChatGroq](https://api.python.langchain.com/en/latest/chat_models/langchain_groq.chat_models.ChatGroq.html) como campos de dicionário válidos."
        st.markdown("Configuração das camadas do agente", help=tooltip)
        new_layer_agent_config = st_ace(
            value=json.dumps(st.session_state.layer_agent_config, indent=2),
            language='json',
            placeholder="Configuração das camadas do agente (JSON)",
            show_gutter=False,
            wrap=True,
            auto_update=True
        )

        if st.form_submit_button("Atualizar Configuração"):
            try:
                new_layer_config = json.loads(new_layer_agent_config)
                set_moa_agent(
                    main_model=new_main_model,
                    cycles=new_cycles,
                    layer_agent_config=new_layer_config,
                    main_model_temperature=main_temperature,
                    override=True
                )
                st.session_state.messages = []
                st.success("Configuração atualizada com sucesso!")
            except json.JSONDecodeError:
                st.error("JSON inválido na configuração do agente de camada. Por favor, verifique sua entrada.")
            except Exception as e:
                st.error(f"Erro ao atualizar a configuração {str(e)}")

    st.markdown("---")
    st.markdown("""
    ### Créditos
    - MOA: [Together AI](https://www.together.ai/blog/together-moa)
    - LLMs: [Groq](https://groq.com/)
    - Paper: [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)
    """)

# Main app layout
st.header("Mistura de Agentes", anchor=False)
st.write("Uma demonstração da arquitetura Mistura de Agentes proposta pela Together AI, alimentada pelos Groq LLMs.")


# Inicializa a variável de estado na sessão
if 'mostrar_imagem' not in st.session_state:
    st.session_state.mostrar_imagem = False

# Função para alternar o estado
def toggle_imagem():
    st.session_state.mostrar_imagem = not st.session_state.mostrar_imagem

# Botão para alternar a exibição da imagem, com texto dinâmico
button_text = 'Ocultar' if st.session_state.mostrar_imagem else 'Veja o Fluxograma'
st.button(button_text, on_click=toggle_imagem)

# Exibe a imagem com base no estado atual
if st.session_state.mostrar_imagem:
    st.image("./static/moa_groq.svg", caption="Fluxo Mistura de Agentes", width=1000)

# Display current configuration
with st.expander("Configuração MOA atual", expanded=False):
    st.markdown(f"**Modelo Principal**: ``{st.session_state.main_model}``")
    st.markdown(f"**Temperatura do Modelo Principal**: ``{st.session_state.main_temp:.1f}``")
    st.markdown(f"**Camadas**: ``{st.session_state.cycles}``")
    st.markdown(f"**Configuração das camadas dos agentes**:")
    new_layer_agent_config = st_ace(
        value=json.dumps(st.session_state.layer_agent_config, indent=2),
        language='json',
        placeholder="Configuração das camadas dos agentes(JSON)",
        show_gutter=False,
        wrap=True,
        readonly=True,
        auto_update=True
    )


# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Faça uma pergunta"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    moa_agent: MOAgent = st.session_state.moa_agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        ast_mess = stream_response(moa_agent.chat(query, output_format='json'))
        response = st.write_stream(ast_mess)
    
    st.session_state.messages.append({"role": "assistant", "content": response})