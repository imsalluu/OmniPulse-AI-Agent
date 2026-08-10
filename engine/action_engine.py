import os
import json
import random
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- TOOL SCHEMAS FOR OPENAI FUNCTION CALLING ---
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate_insurance_quote",
            "description": "Calculates monthly and annual insurance premium quotes dynamically based on customer age, coverage amount, and product type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "Age of the customer in years."
                    },
                    "coverage_amount": {
                        "type": "number",
                        "description": "Desired total insurance coverage amount in USD (e.g. 250000)."
                    },
                    "product_type": {
                        "type": "string",
                        "enum": ["Term Life", "Whole Life", "Health", "Universal Life"],
                        "description": "Type of insurance policy requested."
                    },
                    "term_years": {
                        "type": "integer",
                        "description": "Duration of policy term in years (optional, default 20)."
                    }
                },
                "required": ["age", "coverage_amount", "product_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_policy_document_sms",
            "description": "Dispatches an SMS/Email containing a policy brochure, quote summary, or terms to the customer's phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Customer phone number or email or name."
                    },
                    "document_type": {
                        "type": "string",
                        "enum": ["Quote Summary", "Policy Brochure", "Coverage Options", "Terms & Conditions"],
                        "description": "Type of document to send."
                    }
                },
                "required": ["recipient", "document_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_customer_crm_profile",
            "description": "Retrieves customer risk tier, policy history, and lifetime value from the CRM database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {
                        "type": "string",
                        "description": "Full name of the customer."
                    }
                },
                "required": ["customer_name"]
            }
        }
    }
]


class ActionEngine:
    """
    Agent Action Triggers Engine:
    Evaluates customer transcript for actionable intents and executes corresponding tools via OpenAI Function Calling.
    """
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=openai_key) if openai_key else None

    # --- INDIVIDUAL TOOL EXECUTORS ---
    def _execute_calculate_insurance_quote(self, age: int, coverage_amount: float, product_type: str, term_years: int = 20) -> dict:
        # Realistic premium calculation algorithm
        base_rate = 0.0015
        if age > 40:
            base_rate += (age - 40) * 0.00008
        if product_type == "Whole Life":
            base_rate *= 2.2
        elif product_type == "Health":
            base_rate *= 1.8
            
        annual_premium = round(coverage_amount * base_rate, 2)
        monthly_premium = round(annual_premium / 12.0, 2)
        quote_id = f"QT-{random.randint(10000, 99999)}"

        return {
            "quote_id": quote_id,
            "product_type": product_type,
            "coverage_amount": f"${coverage_amount:,.2f}",
            "term_years": term_years,
            "monthly_premium": f"${monthly_premium:,.2f}",
            "annual_premium": f"${annual_premium:,.2f}",
            "estimated_for_age": age
        }

    def _execute_send_policy_document_sms(self, recipient: str, document_type: str) -> dict:
        return {
            "status": "sent",
            "document_type": document_type,
            "recipient": recipient,
            "timestamp": datetime.now().isoformat(),
            "confirmation_code": f"SMS-{random.randint(100000, 999999)}"
        }

    def _execute_fetch_customer_crm_profile(self, customer_name: str) -> dict:
        tiers = ["Preferred VIP", "Standard", "High Value"]
        return {
            "customer_name": customer_name,
            "crm_id": f"CRM-{random.randint(1000, 9999)}",
            "risk_tier": random.choice(tiers),
            "existing_policies": ["Auto Protect 2023"],
            "loyalty_years": random.randint(1, 5)
        }

    async def eval_and_execute_actions(self, cleaned_text: str, entities: dict = None) -> list:
        """
        Evaluates customer speech using OpenAI Function Calling.
        Executes triggered tools and returns structured execution results.
        """
        if not self.client:
            return []

        prompt = (
            f"Analyze this customer statement from a live call: '{cleaned_text}'.\n"
            f"Extracted Entities: {json.dumps(entities or {})}.\n"
            f"If the customer is asking for a price/quote, wanting a document/brochure/SMS sent, "
            f"or mentioning their name for history lookup, call the appropriate tool."
        )

        executed_actions = []

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an automated Action Dispatcher for an Insurance AI Co-pilot."},
                    {"role": "user", "content": prompt}
                ],
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0
            )

            message = response.choices[0].message

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception:
                        args = {}

                    result = None
                    if fn_name == "calculate_insurance_quote":
                        result = self._execute_calculate_insurance_quote(
                            age=int(args.get("age", 35)),
                            coverage_amount=float(args.get("coverage_amount", 250000)),
                            product_type=args.get("product_type", "Term Life"),
                            term_years=int(args.get("term_years", 20))
                        )
                    elif fn_name == "send_policy_document_sms":
                        result = self._execute_send_policy_document_sms(
                            recipient=str(args.get("recipient", "Customer")),
                            document_type=str(args.get("document_type", "Quote Summary"))
                        )
                    elif fn_name == "fetch_customer_crm_profile":
                        result = self._execute_fetch_customer_crm_profile(
                            customer_name=str(args.get("customer_name", "Valued Customer"))
                        )

                    if result:
                        executed_actions.append({
                            "tool": fn_name,
                            "arguments": args,
                            "status": "success",
                            "result": result
                        })

        except Exception as e:
            print(f"[ACTION ENGINE ERROR] {e}")

        return executed_actions
