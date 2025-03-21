# RAG example

##　执行步骤

1. 下载依赖

   配置python解析器。

   ```shell
   pip install -r requirements.txt
   ```

2. 执行程序

   ```shell
   cd RAG-quickStart
   python rag.py
   ```

   如果执行未成功，请使用https://github.com/chatanywhere/GPT_API_free?tab=readme-ov-file 网站获取免费API key并更新代码中OPENAI_API_KEY环环境变量。

##　预期结果

```shell
(.venv) PS D:\RAG-quickStart\RAG-quickStart> python rag.py
USER_AGENT environment variable not set, consider setting it to identify your requests.
D:\RAG-quickStart\.venv\Lib\site-packages\langsmith\client.py:277: LangSmithMissingAPIKeyWarning: API key must be provided when using hosted LangSmith API
  warnings.warn(
Task decomposition involves breaking down complex tasks into smaller and simpler steps to make them more manageable. This process can be done using techniques like Chain of Thought or Tree of Thoughts, which help the model think step by step and explore multiple reasoning possibilities. Task decomposition can be achieved through simple prompting, task-specific instructions, or human inputs.

```

