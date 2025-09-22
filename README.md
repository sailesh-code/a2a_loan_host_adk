# A2A Consumer Banking System

A sophisticated agent-to-agent (A2A) banking system that enables automated corporate credit line processing with ESG (Environmental, Social, and Governance) integration. The system simulates multiple competing banks processing credit applications through AI agents built with Google's Agent Development Kit (ADK) and the A2A protocol framework.

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Banking Policies](#banking-policies)
- [ESG Integration](#esg-integration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

The A2A Consumer Banking System is a proof-of-concept implementation demonstrating how financial institutions can automate corporate credit line processing using AI agents. The system includes five fully implemented bank agents and a consumer host agent with web interface:

**Bank Agents:**
- **CloudTrust Financial Agent** (Port 10002) - Conservative lending with strong ESG focus
- **Finovate Bank Agent** (Port 10003) - Competitive lending with ESG integration
- **Zentra Bank Agent** (Port 10004) - Specialized lending solutions
- **NexVault Bank Agent** (Port 10005) - Digital-first banking approach
- **Byte Bank Agent** (Port 10006) - Technology-focused lending

**Consumer Agent:**
- **Host Agent with Web UI** - Interactive web interface for submitting credit applications and analyzing loan offers from multiple banks

Each bank agent processes credit applications through a comprehensive workflow including risk assessment, ESG evaluation, interest rate calculation, and loan offer generation. The system includes pre-configured company profiles and sample credit requests for testing and demonstration purposes.

## Technology Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Google ADK (Agent Development Kit)** - AI agent framework
- **A2A SDK** - Agent-to-agent communication protocol
- **Flask** - Web framework for HTTP endpoints
- **Starlette** - ASGI framework for async operations
- **Uvicorn** - ASGI server implementation

### AI and ML
- **Google Generative AI** - LLM integration (Gemini 2.5 Flash)
- **Tachyon ADK Client** - AI model client for agent operations

### Data and Serialization
- **Pydantic** - Data validation and serialization
- **JSONSchema** - API schema validation
- **dataclasses-json** - Python dataclass JSON serialization

### Networking and Security
- **httpx** - Async HTTP client
- **requests** - HTTP client for ESG data fetching
- **cryptography** - Cryptographic operations
- **PyJWT** - JSON Web Token implementation

### Development and Testing
- **pytest** - Testing framework
- **pytest-asyncio** - Async testing support
- **black** - Code formatting
- **python-dotenv** - Environment variable management

## Key Features

### 🏦 Multi-Bank Processing
- Multiple competing bank agents with different lending policies
- Real-time credit application processing
- Automated risk assessment and loan offer generation

### 🌱 ESG Integration
- Comprehensive ESG scoring based on:
  - Carbon emissions intensity analysis
  - Industry benchmarking
  - ESG certifications (B-Corp, ISO 14001, SBTI, etc.)
  - Environmental performance metrics
- Interest rate discounts for strong ESG performers
- Real-time ESG report fetching from external URLs

### 💰 Sophisticated Risk Assessment
- Multi-factor risk scoring including:
  - Financial health metrics (profitability, leverage, scale)
  - Industry risk classifications
  - Geographic/jurisdiction compliance
  - Debt-to-asset ratio analysis
- Dynamic lending ratio calculations
- Weighted risk score methodology

### 🔄 Loan Negotiation
- Interactive loan offer negotiation
- ESG-based interest rate adjustments
- Automated counter-offer generation

### 📋 Comprehensive Compliance
- Industry eligibility checks
- Jurisdiction risk assessment
- Regulatory compliance verification
- Anti-money laundering (AML) screening capabilities

### 🔗 Agent-to-Agent Communication
- Wells Fargo Agent Protocol (WFAP) implementation
- JSON-RPC message handling
- Digital signature verification
- Structured data exchange

## Project Structure

```
a2a_consumer_bank/
├── artifacts/
│   └── arch/
│       └── schema.json                    # WFAP protocol schema definitions
├── code/
│   ├── src/                               # Main source code directory
│   │   ├── bank_agent_1_adk/              # CloudTrust Financial Agent (Port 10002)
│   │   │   ├── __init__.py
│   │   │   ├── main.py                    # Server entry point
│   │   │   ├── agent.py                   # Agent configuration and workflow
│   │   │   ├── agent_executor.py          # Agent execution logic
│   │   │   ├── bank_policy_tools.py       # Risk assessment and loan tools
│   │   │   └── wfap_protocol.py           # WFAP protocol implementation
│   │   ├── bank_agent_2_adk/              # Finovate Bank Agent (Port 10003)
│   │   │   ├── __init__.py
│   │   │   ├── main.py                    # Server entry point
│   │   │   ├── agent.py                   # Agent configuration and workflow
│   │   │   ├── agent_executor.py          # Agent execution logic
│   │   │   ├── bank_policy_tools.py       # Risk assessment and loan tools
│   │   │   └── wfap_protocol.py           # WFAP protocol implementation
│   │   ├── bank_agent_3_adk/              # Zentra Bank Agent (Port 10004)
│   │   │   ├── __init__.py
│   │   │   ├── main.py                    # Server entry point
│   │   │   ├── agent.py                   # Agent configuration and workflow
│   │   │   ├── agent_executor.py          # Agent execution logic
│   │   │   ├── bank_policy_tools.py       # Risk assessment and loan tools
│   │   │   └── wfap_protocol.py           # WFAP protocol implementation
│   │   ├── bank_agent_4_adk/              # NexVault Bank Agent (Port 10005)
│   │   │   ├── __init__.py
│   │   │   ├── main.py                    # Server entry point
│   │   │   ├── agent.py                   # Agent configuration and workflow
│   │   │   ├── agent_executor.py          # Agent execution logic
│   │   │   ├── bank_policy_tools.py       # Risk assessment and loan tools
│   │   │   └── wfap_protocol.py           # WFAP protocol implementation
│   │   ├── bank_agent_5_adk/              # Byte Bank Agent (Port 10006)
│   │   │   ├── __init__.py
│   │   │   ├── main.py                    # Server entry point
│   │   │   ├── agent.py                   # Agent configuration and workflow
│   │   │   ├── agent_executor.py          # Agent execution logic
│   │   │   ├── bank_policy_tools.py       # Risk assessment and loan tools
│   │   │   └── wfap_protocol.py           # WFAP protocol implementation
│   │   ├── host_agent_adk/                # Consumer/Host Agent with Web UI
│   │   │   ├── __init__.py
│   │   │   ├── agent.py                   # Consumer agent logic
│   │   │   ├── company_config.py          # Company configuration templates
│   │   │   ├── host_tools.py              # Host agent tools and utilities
│   │   │   ├── loan_offer_analyzer_tool.py # Loan offer analysis capabilities
│   │   │   ├── remote_agent_connection.py # Bank agent communication
│   │   │   ├── web_server.py              # Flask web server
│   │   │   ├── start_custom_ui.py         # Web UI startup script
│   │   │   ├── index.html                 # Web interface HTML
│   │   │   ├── styles.css                 # Web interface styling
│   │   │   └── script.js                  # Web interface JavaScript
│   │   ├── requirements.txt               # Python dependencies
│   │   ├── start_adk_web_server.bat      # ADK web server startup script
│   │   └── start_banking_system.bat      # Full system startup script
│   └── test/                             # Test data and configurations
│       ├── company_configs/              # Pre-configured company profiles
│       │   ├── __init__.py
│       │   ├── healthcare_services_config.py
│       │   ├── manufacturing_company_config.py
│       │   ├── renewable_energy_mnc_config.py
│       │   ├── retail_chain_config.py
│       │   ├── startup_tech_config.py
│       │   └── technology_services_config.py
│       └── credit_requests/              # Sample credit request JSON files
│           ├── cloudscale_ai_loan_requests.json
│           ├── greenpower_global_loan_requests.json
│           ├── midwest_steel_loan_requests.json
│           ├── pacific_coast_loan_requests.json
│           └── sunshine_medical_loan_requests.json
├── Wells Fargo Agent Protocol.docx       # Protocol documentation
└── README.md                             # This file
```

### Key Files

- **`main.py`** - Starlette/Uvicorn server with JSON-RPC handling
- **`agent.py`** - ADK agent configuration with comprehensive loan processing workflow
- **`bank_policy_tools.py`** - Core banking functions (risk assessment, ESG scoring, loan calculations)
- **`wfap_protocol.py`** - Data models and protocol definitions
- **`schema.json`** - JSON schema for WFAP protocol validation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd a2a_consumer_bank
   ```

2. **Navigate to the source directory**
   ```bash
   cd code/src
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import a2a, google.adk; print('Dependencies installed successfully')"
   ```

## Configuration

### Environment Variables

Create a `.env` file in the `code/src` directory with the following variables:

```bash
# Google AI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Bank Configuration
BANK_1_NAME=CloudTrust Financial
BANK_1_PORT=10002
BANK_2_NAME=Finovate Bank
BANK_2_PORT=10003

# Logging
LOG_LEVEL=INFO

# Optional: ESG Data Sources
ESG_REPORT_TIMEOUT=10
```

### Banking Policies

Each bank agent uses configurable policies defined in `wfap_protocol.py`:

- **Minimum Requirements**: Revenue thresholds, credit scores, business age
- **Risk Classifications**: Industry risk levels, geographic restrictions
- **Interest Rates**: Base rates, risk premiums, ESG discounts
- **Loan Terms**: Duration limits by purpose, collateral requirements

## Usage

### Starting the System

#### Option 1: Full System Startup (Recommended)
```bash
cd code/src
start_banking_system.bat
```

This will start:
- Bank Agent 1 (CloudTrust Financial) on port 10002
- Bank Agent 2 (Finovate Bank) on port 10003
- Bank Agent 3 (Zentra Bank) on port 10004
- Bank Agent 4 (NexVault Bank) on port 10005
- Bank Agent 5 (Byte Bank) on port 10006
- Host Agent with Custom Web UI

#### Option 2: With ADK Web Interface
```bash
cd code/src
start_adk_web_server.bat
```

This starts all bank agents plus the ADK web interface for agent management.

#### Option 3: Individual Components
```bash
# Start individual bank agents
cd code/src
python -m bank_agent_1_adk.main  # CloudTrust Financial (Port 10002)
python -m bank_agent_2_adk.main  # Finovate Bank (Port 10003)
python -m bank_agent_3_adk.main  # Zentra Bank (Port 10004)
python -m bank_agent_4_adk.main  # NexVault Bank (Port 10005)
python -m bank_agent_5_adk.main  # Byte Bank (Port 10006)

# Start host agent with web UI
python -m host_agent_adk.start_custom_ui
```

#### Option 4: Web UI Only (for testing with sample data)
```bash
cd code/src
python -m host_agent_adk.start_custom_ui
```

This starts the consumer web interface with pre-configured company profiles and sample credit requests.

### Using Pre-configured Test Data

The system includes comprehensive test data for easy demonstration:

#### Company Configurations (`code/test/company_configs/`)
- **Healthcare Services** - Medical services company profile
- **Manufacturing Company** - Industrial manufacturing profile  
- **Renewable Energy MNC** - Clean energy multinational profile
- **Retail Chain** - Multi-location retail business profile
- **Startup Tech** - Early-stage technology company profile
- **Technology Services** - Established tech services profile

#### Sample Credit Requests (`code/test/credit_requests/`)
- **CloudScale AI** - AI/ML company requesting working capital and equipment financing
- **GreenPower Global** - Renewable energy company credit applications
- **Midwest Steel** - Manufacturing company loan requests
- **Pacific Coast** - Coastal business credit applications
- **Sunshine Medical** - Healthcare services financing requests

### Sending Credit Applications

#### Using the Web Interface
1. Start the system using Option 1 or 4 above
2. Open your browser to the host agent web interface
3. Select a pre-configured company profile or create custom application
4. Submit to one or all bank agents
5. Compare loan offers and terms

#### JSON-RPC Request Format (Direct API)
```bash
curl -X POST http://localhost:10002/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "send_message",
    "params": {
      "message": {
        "taskId": "task_123",
        "contextId": "context_456",
        "parts": [{
          "type": "text",
          "text": "{\"intent_id\": \"req_001\", \"sender_name\": \"TechCorp Inc\", \"amount_value\": 500000, \"repayment_duration\": 24, \"purpose\": \"working capital\", \"industry_code\": \"541\", \"jurisdiction\": \"US\", \"financials_annual_revenue\": 10000000, \"financials_net_income\": 1500000, \"financials_assets_total\": 8000000, \"financials_liabilities_total\": 3000000, \"esg_certifications\": \"B-Corp,ISO14001\", \"esg_reporting_url\": \"https://example.com/esg-report\", \"carbon_emissions\": 450}"
        }]
      }
    },
    "id": 1
  }'
```

#### Required Application Fields
- `intent_id`: Unique application identifier
- `sender_name`: Company name
- `amount_value`: Requested credit amount (USD)
- `repayment_duration`: Desired term in months
- `purpose`: Loan purpose (working capital, equipment, etc.)
- `industry_code`: NAICS industry code
- `jurisdiction`: Company jurisdiction (US, CA, UK, etc.)
- `financials_annual_revenue`: Annual revenue (USD)
- `financials_net_income`: Net income (USD)
- `financials_assets_total`: Total assets (USD)
- `financials_liabilities_total`: Total liabilities (USD)

#### Optional ESG Fields
- `esg_certifications`: Comma-separated certifications
- `esg_reporting_url`: URL to ESG report
- `carbon_emissions`: Annual CO2 emissions (tons)

### Response Format

#### Successful Loan Offer
```json
{
  "offer_id": "uuid-generated",
  "intent_id": "req_001",
  "created_at": "2024-01-15T12:00:00Z",
  "protocol_version": "WFAP-1.0",
  "bank_agent_id": "WF-BANK-AGENT-001",
  "status": "OFFER_EXTENDED",
  "amount_approved": 500000,
  "currency": "USD",
  "interest_rate_annual": 7.25,
  "repayment_duration_months": 24,
  "repayment_schedule": "amortizing",
  "esg_impact_summary": "TechCorp demonstrates strong environmental commitment through B-Corp certification and carbon performance 25% better than industry average."
}
```

#### Loan Rejection
```json
{
  "status": "REJECTED",
  "intent_id": "req_001",
  "rejection_reason": "Annual revenue $2,000,000 is below minimum $5,000,000",
  "detailed_results": [...]
}
```

## API Reference

### Bank Agent Endpoints

#### POST `/`
Main JSON-RPC endpoint for credit applications.

**Method**: `send_message`

**Parameters**:
- `message.parts[0].text`: JSON string containing credit application

**Response**: Loan offer JSON or error message

#### GET `/.well-known/agent-card.json`
Returns agent capabilities and metadata.

### Core Banking Functions

#### `perform_initial_risk_assessment()`
Conducts comprehensive risk assessment including:
- Industry eligibility verification
- Loan amount limit checks
- Jurisdiction compliance
- Financial health analysis

#### `calculate_interest_rate_and_offer()`
Calculates interest rates using:
- Base rate + risk premium - ESG discount
- Weighted risk scoring (profitability, leverage, scale)
- Industry benchmarking
- ESG performance evaluation

#### `calculate_final_approved_amount()`
Determines approved amount using:
- Dynamic lending ratio model
- Revenue-based calculations
- Risk and ESG adjustments
- Policy limit enforcement

#### `calculate_approved_repayment_duration()`
Sets repayment terms based on:
- Loan purpose duration limits
- Risk-adjusted maximums
- Policy compliance requirements

## Banking Policies

### Risk Classifications

#### Industry Risk Levels
- **Low Risk**: Healthcare, utilities, food processing, professional services, technology
- **Medium Risk**: Manufacturing, retail, transportation, construction, wholesale
- **High Risk**: Oil & gas, mining, agriculture, hospitality, real estate
- **Prohibited**: Cryptocurrency, gambling, adult entertainment, cannabis

#### Geographic Risk
- **Acceptable**: US, Canada, UK, Germany, France, Australia, Japan
- **High Risk**: Countries under sanctions or with high regulatory risk

### Financial Requirements

#### Minimum Eligibility (Conservative Bank Policy)
- Annual Revenue: $5,000,000+
- Years in Business: 5+
- Credit Score: 700+
- Debt Service Coverage Ratio: 1.5+
- Current Ratio: 1.5+
- Maximum Debt-to-Equity: 2.5

#### Credit Limits
- Minimum: $250,000
- Maximum: $5,000,000
- Typical: 10% of annual revenue

## ESG Integration

### ESG Scoring Methodology

#### Carbon Performance (70% weight)
- Industry-specific emissions benchmarking
- Emissions intensity calculation (tons CO2e per $M revenue)
- Performance categories from "> 50% Better" to "> 20% Worse"

#### Qualitative Assessment (30% weight)
- **B-Corp Certification**: 40 points
- **ISO 14001**: 25 points
- **Science Based Targets Initiative**: 25 points
- **LEED Certification**: 15 points
- **Carbon Neutral**: 20 points
- **Additional certifications**: 10-20 points each

### ESG Discount Structure
- **ESG Leader (90-100 score)**: 0.75% rate discount
- **Strong Performer (75-89)**: 0.50% rate discount
- **Average Performer (50-74)**: 0.25% rate discount
- **Laggard (< 50)**: No discount

### Industry Carbon Benchmarks
```
Software Development: 10 tons CO2e/$M revenue
Manufacturing: 180 tons CO2e/$M revenue
Food Processing: 250 tons CO2e/$M revenue
Utilities: 400 tons CO2e/$M revenue
Oil & Gas: 600 tons CO2e/$M revenue
```

## Development

### Running Tests
```bash
cd code/src
pytest
```

### Code Formatting
```bash
black .
```

### Adding New Bank Agents

1. **Create new agent directory**
   ```bash
   mkdir bank_agent_3_adk
   cd bank_agent_3_adk
   ```

2. **Copy template files**
   ```bash
   cp ../bank_agent_1_adk/*.py .
   ```

3. **Update configuration**
   - Modify `main.py` port number
   - Update agent name and description
   - Adjust banking policies in `wfap_protocol.py`

4. **Update startup scripts**
   Add new agent to `start_banking_system.bat`

### Extending ESG Capabilities

1. **Add new certification types** in `bank_policy_tools.py`:
   ```python
   certification_points = {
       "NEW_CERT": 30,  # Points for new certification
       ...
   }
   ```

2. **Update industry benchmarks**:
   ```python
   industry_benchmarks = {
       "999": 150,  # New NAICS code benchmark
       ...
   }
   ```

3. **Modify scoring algorithms** in ESG calculation functions

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
netstat -ano | findstr :10002
# Kill process
taskkill /PID <process_id> /F
```

#### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

#### ESG Report Fetch Failures
- Verify ESG report URL accessibility
- Check network connectivity
- Review timeout settings in environment variables

#### Agent Startup Failures
- Verify Google API key configuration
- Check Python version compatibility (3.8+)
- Review log output for specific error messages

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python -m bank_agent_1_adk.main
```

### Health Checks

Verify all bank agents are running:
```bash
# Check all bank agents
curl http://localhost:10002/.well-known/agent-card.json  # CloudTrust Financial
curl http://localhost:10003/.well-known/agent-card.json  # Finovate Bank  
curl http://localhost:10004/.well-known/agent-card.json  # Zentra Bank
curl http://localhost:10005/.well-known/agent-card.json  # NexVault Bank
curl http://localhost:10006/.well-known/agent-card.json  # Byte Bank
```

Or check programmatically using the host agent:
```bash
cd code/src
python -c "from host_agent_adk.start_custom_ui import check_bank_agents; import asyncio; asyncio.run(check_bank_agents())"
```

## Contributing

### Development Guidelines

1. **Code Style**: Follow PEP 8 and use Black for formatting
2. **Testing**: Write tests for new features
3. **Documentation**: Update README for significant changes
4. **Commit Messages**: Use conventional commit format

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Reporting Issues

Please include:
- Python version
- Operating system
- Error messages and stack traces
- Steps to reproduce
- Expected vs actual behavior

---

**Note**: This is a proof-of-concept system for demonstration purposes. For production use, additional security measures, comprehensive testing, and regulatory compliance reviews would be required.
