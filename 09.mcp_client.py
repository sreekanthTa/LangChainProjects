from fastmcp import Client
import asyncio

async def main():

    client = Client("http://localhost:8001/mcp")

    async with client:   # 🔥 REQUIRED

        tools = await client.list_tools()
        # print("Available tools:", tools)

        result = await client.call_tool(
            "add_numbers",
            {"a": 5, "b": 7}
        )

        print("Result:", result.content[0].text)

asyncio.run(main())