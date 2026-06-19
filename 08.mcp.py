from fastmcp import FastMCP

mcp = FastMCP("practice")


# Define a resource
@mcp.resource("config://app")
def app_config() -> str:
    return """
    {
        "version": "1.0.0",
        "environment": "development"
    }
    """

# Define a prompt
@mcp.prompt
def summarize(text: str) -> str:
    return f"Summarize the following text:\n\n{text}"

@mcp.tool(app=True)
def echo(text: str) -> str:
	"Return the input text unchanged."
	return text


@mcp.tool(app=True)
def add_numbers(a: float, b: float) -> float:
	"Return the sum of two numbers."
	return a + b


@mcp.tool(app=True)
def get_datetime() -> str:
	"Return current UTC datetime in ISO 8601 format."
	from datetime import datetime
	return datetime.utcnow().isoformat() + "Z"


if __name__ == "__main__":
	# Simple local demo when running the file directly
    # mcp.run()
    mcp.run(
            transport="http", host="0.0.0.0", port=8001, path="/mcp"
        )