import rag as Rag



# if __name__ == "__main__":
#     rag = Rag.RAG()
#     rag.build_data_rag()
#     rag.build_pipeline_rag()
#     rag.chat_with_rag()


from streamlit.web import cli as stcli
import sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "src/front/app.py"]
    sys.exit(stcli.main())

