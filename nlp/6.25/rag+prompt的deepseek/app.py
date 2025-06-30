import gradio as gr
from rag_core import rag_chat
iface = gr.Interface(
    fn=rag_chat,
    inputs=gr.Textbox(label="请输入你的问题"),
    outputs=gr.Textbox(label="RAG问答结果"),
    title="本地RAG + DeepSeek-1.5B 中文知识问答",
    description="结合检索增强和本地大模型DeepSeek进行中文问答。"
)

iface.launch(server_name="0.0.0.0", server_port=7860)