# Smart Receipt Assistant

Smart Receipt Assistant is a multi-agent system designed to extract, normalize, store and query information from retail receipts.  It combines an orchestrator (Complex LangGraph) with a remote A2A agent (ADK Agent) 
to perform OCR on receipt images, convert currencies, translate text and provide relevant information from external knowledge bases. The project also exposes a FastAPI backend and a Gradio front-end for end-user interaction.

# Complex Langgraph 

The complex langgraph orchestrates  different agents via a connected graph structure 

![Alt text](https://github.com/christinasaidy/inMind_Final_Project/blob/new_start/assets/graph.png)

- OCR AGENT: This node is the remote A2A agent that extracts text from receipt images. [Implementation](#ADK-OCR-Agent-+-A2A)
  
- NORMALIZE AGENT: Takes the extracted receipt text. This agent has the responsibility to extract important information from the text like category, date, and transforms the text into JSON format that matches the database schema. This is achieved by using `llm_with_structured_output()` with a predefined class to make sure th model is outputting the correct shape. This agent needs to infer two schemas: receipts & receipt items

- STORE AGENT: This agent takes the normalized receipt and inserts it into an SQL database using *sqlalchemy*. There are two tables: receipts & receipt item.

- QUERY AGENT: This agent is a smart sql agent that utilizes *langchain's SQLtoolkit* to retrieve database information. When a user asks any question related to a receipt they previously provided, this agent looks at the database's columns, tables, schema and deduces an SQL query to execute to correctly answer the user.

- RAG NODE: This node mixes two types of agentic rag implmentations, the first rag is about a bunch of articles having to do with lebanon's economic state. The second rag is an SQL database with information about product prices in Lebanon. This agent should correctly answer user inqueries about these topics

- ROUTER/SUPERVISOR : This node is the orchestrator behind all the above nodes. It's sole responsibility is routing to the correct node based on user input.

All these agents use gemini flash models for simplicity of tasks.

### Challenges with implementing the Complex Langgraph

All of the nodes when done executing go back to the router agent to make the next decision. This was the main challenge in my implementation. The first step towards achieving this was forcing the model to adhere to a specific output shape using  `llm_with_structured_output(RouteDecision)`. This decision class tells the model that it has pick a next node to route to, and explain why (explanantion mainly for debugging). Also it has the respoonsibility to extract important info from user's input like receipt_uri or the rag question. After that, it was easy to update the states according to the the llm's structured ouput. For example the next state would be: `state["next"] = decision.next` (decision being the structured llm output)

The next step towards making sure the model would route correctly was passing it a summary of the current states so its able to better make decisions.

The structured output and summary made things easier, but the key to making sure the model would work correctly was the the prompt. I unfortunately didnt keep track of the the prompt changes. However my current prompt explicitly asks the model to follow a certain workflow: First: OCR agent, Second: Normalize agent, Third: Store agent. And if the user is asking about receipt info or rag related info, it should alsp route accordingly. This was a long and tedious process of trial and error. In the original attempts, the model would keep routing to the same mutliple times even after it executed its function. This resulted in the waste of alot of takens and wasting free api calls. 

A separate challenge i faced was that intially i wanted to connect my router agent to my rag mcp_server, so instead of introducing a new agent to answer rag questions, the router would use the tools to do that. I loaded the tools and binded them to the model with `llm.bind_tools()`, however my previously implemenetd `llm_with_structured_output()` stopped working. As it turns out, the two methods arent compatible. The solution: Make the rag entity its own agent and leave the router just for routing.

The smart sql agent wasn't so smart in the first attempts. It was instructed to use the SQLToolkit to check out database schema and table information before executing a query. However, a lot of the times it would still ask the user to pick which table in the database to query from. Another issue is how case sensitive sql queries are. So if the user asked how much did i spend in *Star Bucks* instead of *Starbucks*, model would tell the user that star bucks data doesnt exist. The solution? *drum roll* 🥁🥁🥁 Fixing the prompt!
The model was instructed never to ask the user for table information, to try more than one query on different tables, and to never perform any other query than *SELECT Queries*. The prompt that made the most improvement was: 

 `When querying for an item or a specific vendor or a specific category the name may vary: "
        "FIRST run: SELECT DISTINCT Item FROM receipts or receipt_items "
        "WHERE lower(Item) LIKE '%milk%'; "
        "Then choose the best match and run a precise SELECT.`
        
This fixed the issue of the user typing a slighty different name than whats stored in the db

### Langgraph Tools Exploration: 

An initial problem with my graph implementation was using  the `goto command` in langgraph for routing as well as also using static edges. This was a logical error in my code because the goto command aims to replace static edges so this didnt make any sense having them both.

In my nodes you can also see different methods i tried for tool integration. The store agent, manually invokes each tool call made by the model. It loops over each tool name and manually adds a `ToolMessage`. In the rag node, i explored langgraphs built in `ToolNode` that also invokes the tools using a loop, it defines `ToolMessage` for you . In the query agent, i tried the built in `create_react_agent` that automatatically does tool calls without doing anything manually. All three methods work the same and effectively.


# ADK OCR Agent + A2A:

The **ADK Agent** lives outside the Complex LangGraph and communicates via the **A2A protocol**.  

### Responsibilities
- **OCR**: Uses Azure Computer Vision to extract text from receipt images.  
- **Extraction**: Parses the OCR text to extract the total amount, purchase date, and currency symbol.  
- **Currency conversion**: Calls `exchangerate.host` to convert amounts to USD.  
- **Translation**: Translates non-English text into English where necessary.  
- **Response format**: Returns a clean JSON with the extracted fields.

This agent was built using googles ADK with a prebuilt `llmAgent()`, it has its own mcp server that holds the tools for calling the azure api & the currency api.

Implement A2A and exposing this agent as a remote agent was pretty straight forward. A2A documentations' git repo provides multiple samples of implementing this protocol. The steps taken were:

- Implmenting the Agent executor class that inherits from an abstract `AgentExecutor` class. This class implements thw two methods resonsible for agent execution: `def invoke()` and and `def cancel()`
- Defining the *Agent SKills* and *Agent Cards* that define what the agent can do and where the agent is available. It exposes a an endpoint for the agent revieling all the important information about it.
- Running a server for this Agent with `A2AStarletteApplication`
- Defining a client class in the complex langgraph that is responsible for sending messages and communicating with the remote A2A agent.

It server must be started **before any OCR requests** are made.

# MCP Servers 

In my complex langgraph, three mcp servers were defined with unqiue tools for Query Agent, Store Agent, RAG agent. 












