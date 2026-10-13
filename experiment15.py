
!pip -q install -U langchain-google-genai

import os
import json
from google.colab import userdata, files
from langchain_google_genai import ChatGoogleGenerativeAI

# Set API key
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")

# Upload receipt image
uploaded = files.upload()
image_path = next(iter(uploaded))

# Read and encode image as Base64
import base64
with open(image_path, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

# Initialize vision-capable LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0
)

# Multimodal prompt
prompt = """
Analyze this restaurant receipt image and extract:
- restaurant_name
- date (YYYY-MM-DD if readable)
- items (list of objects with name and price)
- subtotal
- tax
- total

Return ONLY a valid JSON object with these keys.
Use null for unreadable or missing values.
Do not invent information. Prices must be numbers.
"""

# Determine image MIME type
import mimetypes
mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

# Send prompt and image to Gemini
response = llm.invoke([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{image_data}"
            }
        ]
    }
])

# Extract text only
output = response.text
if isinstance(output, list):
    output = "".join(
        part.get("text", "")
        for part in output
        if isinstance(part, dict) and part.get("type") == "text"
    )

# Parse and validate JSON
try:
    data = json.loads(output.strip().replace("```json", "").replace("```", "").strip())

    print("=== EXTRACTED RECEIPT JSON ===")
    print(json.dumps(data, indent=2))

    subtotal = data.get("subtotal")
    tax = data.get("tax")
    total = data.get("total")

    print("\n=== VALIDATION ===")
    if all(isinstance(x, (int, float)) and not isinstance(x, bool)
           for x in (subtotal, tax, total)):
        print("Subtotal + Tax =", round(subtotal + tax, 2))
        print("Receipt Total =", round(total, 2))
        print("Consistent:", abs(subtotal + tax - total) < 0.01)
    else:
        print("Cannot validate: subtotal, tax, or total is missing.")

except json.JSONDecodeError as e:
    print("Invalid JSON:", e)
    print("Raw response:", output)
