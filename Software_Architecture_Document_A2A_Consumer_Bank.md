# Software Architecture Document: A2A Consumer Banking System Financial Agent Network

**Version:** 1.0  
**Date:** September 22, 2025  
**Author:** AI Enterprise Architect

---

## 1. Executive Summary

The A2A Consumer Banking System is a sophisticated agent-to-agent (A2A) financial network that enables automated corporate credit line processing through AI agents. The system demonstrates how financial institutions can automate credit applications, risk assessment, and loan offer generation using the Wells Fargo Agent Protocol (WFAP) for secure inter-agent communication.

The architecture consists of a Consumer Agent (host) that coordinates with multiple competing Bank Agents to process credit requests. Each bank agent applies its own risk assessment policies, ESG evaluation criteria, and interest rate calculations to generate personalized loan offers. The system integrates Environmental, Social, and Governance (ESG) factors into lending decisions, providing interest rate discounts for companies with strong sustainability profiles.

The core business problem this system solves is the automation and standardization of corporate credit applications across multiple financial institutions, while maintaining compliance with regulatory requirements and incorporating modern ESG considerations into lending decisions.

---

## 2. Architectural Overview

The system follows a distributed microservices architecture with the following key components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    A2A Consumer Banking System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐           ┌─────────────────────────────┐  │
│  │   Consumer      │           │        Bank Agents          │  │
│  │   Agent         │◄─────────►│                             │  │
│  │   (Host)        │   WFAP    │  ┌─────┬─────┬─────┬─────┐  │  │
│  │                 │ Protocol  │  │Bank1│Bank2│Bank3│Bank4│  │  │
│  │ - Web UI        │           │  │Port │Port │Port │Port │  │  │
│  │ - Credit Req    │           │  │10002│10003│10004│10005│  │  │
│  │ - Offer Analysis│           │  └─────┴─────┴─────┴─────┘  │  │
│  │ - Negotiation   │           │                             │  │
│  └─────────────────┘           └─────────────────────────────┘  │
│           │                                    │                │
│           │                                    │                │
│  ┌─────────────────┐                 ┌─────────────────────┐   │
│  │  Google ADK     │                 │   Banking Policies  │   │
│  │  - AI Agents    │                 │   - Risk Assessment │   │
│  │  - LLM Models   │                 │   - ESG Scoring     │   │
│  │  - Tool System  │                 │   - Interest Rates  │   │
│  └─────────────────┘                 └─────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Communication Flow:**
1. Consumer Agent receives credit request via web interface
2. Request is broadcast to all available Bank Agents via WFAP protocol
3. Each Bank Agent processes request through risk assessment pipeline
4. Bank Agents return individual loan offers with ESG impact summaries
5. Consumer Agent aggregates offers and presents comparison to user
6. User can negotiate terms with specific banks
7. Final offer selection and acceptance

---

## 3. Web Finance Application Protocol (WFAP) Definition

### 3.1. Protocol Intent and Philosophy

WFAP (Wells Fargo Agent Protocol) is a standardized request/response protocol designed for secure and compliant financial communications between AI agents. The protocol's core philosophy centers on:

- **Security**: All messages are digitally signed and include cryptographic verification
- **Compliance**: Built-in regulatory compliance checks and audit trails
- **Structured Data Exchange**: Standardized JSON schemas for consistent data interpretation
- **ESG Integration**: Native support for Environmental, Social, and Governance factors
- **Interoperability**: Agent-agnostic protocol supporting multiple financial institutions

The protocol facilitates two primary message types:
- **Intent Packets**: Consumer requests for financial products (credit lines, loans)
- **Offer Packets**: Bank responses with terms and conditions

### 3.2. WFAP Message Structure (Markdown Definition)

The protocol defines two primary message types:

* **Intent Packet (Client -> Server):** The message sent by a consumer to request a financial product (e.g., credit line). Contains company information, financial data, credit requirements, and ESG preferences.

* **Offer Packet (Server -> Client):** The message returned by a bank with the terms of a potential offer. Includes loan terms, interest rates, ESG impact assessment, and regulatory compliance information.

### 3.3. Compliance Schemas (JSON Schema)

#### 3.3.1. Intent Packet Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "WFAP Intent Packet",
  "type": "object",
  "properties": {
    "header": {
      "type": "object",
      "properties": {
        "messageId": { "type": "string", "format": "uuid" },
        "timestamp": { "type": "string", "format": "date-time" },
        "protocolVersion": { "type": "string", "const": "WFAP-1.0" }
      },
      "required": ["messageId", "timestamp", "protocolVersion"]
    },
    "consumer": {
      "type": "object",
      "properties": {
        "companyId": { "type": "string" },
        "companyName": { "type": "string" },
        "creditScore": { "type": "number", "minimum": 300, "maximum": 850 },
        "annualRevenue": { "type": "number" },
        "yearsInBusiness": { "type": "integer", "minimum": 0 },
        "industry": { "type": "string" },
        "jurisdiction": { "type": "string" },
        "taxId": { "type": "string" },
        "registrationNumber": { "type": "string" },
        "contactEmail": { "type": "string", "format": "email" }
      },
      "required": ["companyId", "companyName", "annualRevenue", "industry", "jurisdiction"]
    },
    "creditRequest": {
      "type": "object",
      "properties": {
        "amount": { "type": "number", "minimum": 1000 },
        "duration": { "type": "integer", "minimum": 1 },
        "purpose": { "type": "string" },
        "preferredInterestRate": { "type": "number", "minimum": 0 },
        "repaymentPreference": { "type": "string" },
        "collateralDescription": { "type": "string" }
      },
      "required": ["amount", "duration", "purpose"]
    },
    "financials": {
      "type": "object",
      "properties": {
        "annualRevenue": { "type": "number" },
        "netIncome": { "type": "number" },
        "assetsTotal": { "type": "number" },
        "liabilitiesTotal": { "type": "number" }
      },
      "required": ["annualRevenue", "netIncome", "assetsTotal", "liabilitiesTotal"]
    },
    "esgData": {
      "type": "object",
      "properties": {
        "certifications": { "type": "string" },
        "reportingUrl": { "type": "string", "format": "uri" },
        "carbonEmissions": { "type": "number" },
        "esgWeight": { "type": "number", "minimum": 0, "maximum": 1 }
      }
    },
    "compliance": {
      "type": "object",
      "properties": {
        "regulatoryConsent": { "type": "boolean" },
        "dataSharingConsent": { "type": "boolean" },
        "jurisdictionCompliance": { "type": "string" }
      },
      "required": ["regulatoryConsent", "dataSharingConsent"]
    },
    "signature": {
      "type": "string",
      "description": "Digital signature of the entire packet (excluding this field)."
    }
  },
  "required": ["header", "consumer", "creditRequest", "financials", "compliance", "signature"]
}
```

#### 3.3.2. Offer Packet Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "WFAP Offer Packet",
  "type": "object",
  "properties": {
    "header": {
      "type": "object",
      "properties": {
        "offerId": { "type": "string", "format": "uuid" },
        "intentId": { "type": "string", "format": "uuid" },
        "timestamp": { "type": "string", "format": "date-time" },
        "protocolVersion": { "type": "string", "const": "WFAP-1.0" }
      },
      "required": ["offerId", "intentId", "timestamp", "protocolVersion"]
    },
    "bankIdentityToken": {
      "type": "object",
      "properties": {
        "bankId": { "type": "string" },
        "bankName": { "type": "string" },
        "agentId": { "type": "string" },
        "regulatoryLicense": { "type": "string" }
      },
      "required": ["bankId", "bankName", "agentId"]
    },
    "offer": {
      "type": "object",
      "properties": {
        "status": { "type": "string", "enum": ["OFFER_EXTENDED", "REJECTED", "OFFER_EXTENDED_WITH_MODIFICATIONS"] },
        "amountApproved": { "type": "number" },
        "currency": { "type": "string", "default": "USD" },
        "interestRateAnnual": { "type": "number" },
        "repaymentDurationMonths": { "type": "integer" },
        "repaymentSchedule": { "type": "string" },
        "originationFee": { "type": "number", "default": 0 },
        "annualFee": { "type": "number", "default": 0 }
      },
      "required": ["status", "amountApproved", "interestRateAnnual", "repaymentDurationMonths"]
    },
    "riskAssessment": {
      "type": "object",
      "properties": {
        "overallRiskScore": { "type": "number", "minimum": 1, "maximum": 5 },
        "riskCategory": { "type": "string", "enum": ["Excellent", "Good", "Average", "Sub-par", "High-Risk"] },
        "industryRiskLevel": { "type": "string", "enum": ["low", "medium", "high", "very_high"] },
        "financialHealthScore": { "type": "number" }
      },
      "required": ["overallRiskScore", "riskCategory"]
    },
    "esgImpact": {
      "type": "object",
      "properties": {
        "esgScore": { "type": "number", "minimum": 0, "maximum": 100 },
        "esgCategory": { "type": "string", "enum": ["ESG Leader", "ESG Strong Performer", "ESG Average Performer", "ESG Laggard"] },
        "carbonFootprint": { "type": "number" },
        "carbonAdjustedRate": { "type": "number" },
        "esgDiscount": { "type": "number" },
        "esgSummary": { "type": "string" },
        "sustainabilityInitiatives": { "type": "array", "items": { "type": "string" } }
      },
      "required": ["esgScore", "esgCategory", "esgSummary"]
    },
    "signature": {
      "type": "string",
      "description": "Digital signature of the entire packet (excluding this field)."
    }
  },
  "required": ["header", "bankIdentityToken", "offer", "riskAssessment", "esgImpact", "signature"]
}
```

---

## 4. Bank Agent Architecture (WFAP Server)

### 4.1. Core Responsibilities

Each Bank Agent operates as an autonomous WFAP server with the following responsibilities:

1. **Listen for Requests**: Actively monitor JSON-RPC endpoints for incoming WFAP Intent Packets on designated network ports (10002-10006)

2. **Process Intents**: Validate message schemas, verify digital signatures, and extract business data from credit applications

3. **Risk Assessment**: Execute comprehensive risk evaluation including:
   - Industry eligibility verification
   - Financial health analysis
   - Jurisdiction compliance checks
   - Debt-to-asset ratio evaluation

4. **Generate Offers**: Apply proprietary business logic and risk models to calculate:
   - Interest rates (base rate + risk premium - ESG discount)
   - Approved credit amounts (dynamic lending ratio model)
   - Repayment terms (purpose-based duration limits)

5. **ESG Integration**: Utilize LLM services to:
   - Fetch and parse ESG reports from provided URLs
   - Calculate carbon performance scores against industry benchmarks
   - Generate human-readable ESG impact summaries
   - Apply ESG-based interest rate discounts

6. **Return Signed Offers**: Construct and digitally sign WFAP Offer Packets before transmission

### 4.2. Component Breakdown

**API Endpoint Layer:**
- Starlette/Uvicorn ASGI server handling JSON-RPC requests
- Agent card endpoint (/.well-known/agent-card.json) for capability discovery
- Main processing endpoint (/) for credit request handling

**Request Validation Service:**
- Schema validation against WFAP Intent Packet specification
- Digital signature verification using cryptographic libraries
- Input sanitization and data integrity checks

**Risk Assessment Engine:**
- Multi-stage risk evaluation pipeline with configurable policies
- Industry risk classification system (low/medium/high/prohibited)
- Financial metrics calculation (profitability, leverage, scale)
- Weighted risk scoring with customizable weights

**ESG Evaluation Service:**
- Carbon performance analysis with industry-specific benchmarks
- Qualitative ESG scoring based on certifications (B-Corp, ISO14001, SBTI, etc.)
- External ESG report fetching and parsing capabilities
- ESG discount calculation (0.00% - 0.75% based on performance)

**Offer Generation Engine:**
- Dynamic lending ratio model for amount approval
- Purpose-based repayment duration calculations
- Interest rate composition (base + risk premium - ESG discount)
- Loan terms and conditions generation

**LLM Integration Service:**
- Google Generative AI (Gemini 2.5 Flash) integration
- ESG impact summary generation
- Natural language processing for complex financial analysis

**Digital Signing Service:**
- Cryptographic signature generation for offer packets
- Public/private key management
- Signature verification capabilities

---

## 5. Consumer Agent Architecture (WFAP Client)

### 5.1. Core Responsibilities

The Consumer Agent serves as the central coordinator for credit requests with these key functions:

1. **User Interface Management**: Provide web-based interface for credit application submission and offer comparison

2. **Intent Construction**: Build structured WFAP Intent Packets from user input, incorporating:
   - Company financial data
   - Credit requirements
   - ESG preferences and certifications
   - Compliance acknowledgments

3. **Agent Discovery**: Automatically discover available Bank Agents through agent card endpoints

4. **Broadcast and Coordination**: Simultaneously send credit requests to multiple Bank Agents with proper session management

5. **Offer Aggregation**: Collect and validate responses from all Bank Agents within configurable timeouts

6. **Decision Support**: Provide comprehensive offer comparison and recommendation capabilities

7. **Negotiation Facilitation**: Enable interactive negotiation with individual banks for improved terms

8. **Final Selection**: Guide users through offer selection and acceptance process

### 5.2. Component Breakdown

**Web Interface Layer:**
- HTML/CSS/JavaScript frontend for user interaction
- Flask web server for HTTP request handling
- Real-time offer display and comparison tables

**Agent Discovery Service:**
- HTTP client for fetching agent cards from Bank Agent endpoints
- A2A protocol compatibility layer
- Dynamic bank agent registry management

**Intent Builder Module:**
- User input validation and sanitization
- WFAP Intent Packet construction and serialization
- Digital signature generation for outbound requests

**Broadcast Service:**
- Parallel request distribution to multiple Bank Agents
- Session ID management for conversation tracking
- Timeout handling and error recovery

**Response Aggregation Engine:**
- Asynchronous response collection from multiple agents
- Response validation and schema verification
- Error handling for partial failures

**Offer Analysis Tool:**
- Multi-criteria decision analysis for offer comparison
- ESG impact weighting and evaluation
- Interest rate and term optimization recommendations

**Negotiation Manager:**
- Interactive communication with individual Bank Agents
- Counter-offer generation and tracking
- Negotiation history and audit trail

**Decision Logic Engine:**
- Configurable ranking algorithms for offer selection
- Risk-adjusted return calculations
- ESG impact scoring and weighting

---

## 6. Banking Policies and Risk Assessment

### 6.1. Risk Classification Framework

**Industry Risk Levels:**
- **Low Risk**: Healthcare services, utilities, food processing, professional services, technology
- **Medium Risk**: Manufacturing, retail, transportation, construction, wholesale
- **High Risk**: Oil & gas, mining, agriculture, hospitality, real estate
- **Prohibited**: Cryptocurrency, gambling, adult entertainment, cannabis, weapons

**Geographic Risk Assessment:**
- **Acceptable Jurisdictions**: US, Canada, UK, Germany, France, Australia, Japan
- **High-Risk Jurisdictions**: Countries under sanctions or with elevated regulatory risk

### 6.2. Financial Requirements (Conservative Bank Policy)

**Minimum Eligibility Criteria:**
- Annual Revenue: $5,000,000+
- Years in Business: 5+
- Credit Score: 700+
- Debt Service Coverage Ratio: 1.5+
- Current Ratio: 1.5+
- Maximum Debt-to-Equity: 2.5

**Credit Limits:**
- Minimum: $250,000
- Maximum: $5,000,000
- Typical: 10% of annual revenue

### 6.3. Interest Rate Calculation Model

**Base Rate Structure:**
- Base Interest Rate: 6.5% (configurable per bank)
- Risk Premium: 0.5% - 6.0% based on weighted risk score
- ESG Discount: 0.0% - 0.75% based on ESG performance

**Weighted Risk Scoring (1-5 scale):**
- Profitability Score (40% weight): Based on net income margin
- Leverage Score (40% weight): Based on debt-to-asset ratio
- Scale Score (20% weight): Based on annual revenue size

### 6.4. ESG Integration Framework

**Carbon Performance Assessment (70% weight):**
- Industry-specific emissions benchmarking
- Emissions intensity calculation (tons CO2e per $M revenue)
- Performance categories from "> 50% Better" to "> 20% Worse" than industry average

**Qualitative ESG Scoring (30% weight):**
- B-Corp Certification: 40 points
- ISO 14001: 25 points
- Science Based Targets Initiative: 25 points
- LEED Certification: 15 points
- Carbon Neutral: 20 points

**ESG Discount Structure:**
- ESG Leader (90-100 score): 0.75% rate discount
- Strong Performer (75-89): 0.50% rate discount
- Average Performer (50-74): 0.25% rate discount
- Laggard (< 50): No discount

---

## 7. Technology Stack and Implementation

### 7.1. Core Technologies

**Backend Framework:**
- Python 3.8+ as primary programming language
- Starlette ASGI framework for async web services
- Uvicorn ASGI server for production deployment
- Flask for Consumer Agent web interface

**AI and Machine Learning:**
- Google ADK (Agent Development Kit) for AI agent framework
- Google Generative AI (Gemini 2.5 Flash) for LLM integration
- Tachyon ADK Client for AI model interactions

**Data and Serialization:**
- Pydantic for data validation and serialization
- JSONSchema for API schema validation
- dataclasses-json for Python dataclass JSON serialization

**Networking and Security:**
- httpx for async HTTP client operations
- requests for synchronous HTTP operations
- cryptography library for digital signatures
- PyJWT for JSON Web Token implementation

**Agent Communication:**
- A2A SDK for agent-to-agent communication protocol
- Custom WFAP protocol implementation
- JSON-RPC for message handling

### 7.2. Deployment Architecture

**Bank Agents (Ports 10002-10006):**
- CloudTrust Financial Agent (Port 10002) - Conservative lending with ESG focus
- Finovate Bank Agent (Port 10003) - Competitive lending with ESG integration
- Zentra Bank Agent (Port 10004) - Specialized lending solutions
- NexVault Bank Agent (Port 10005) - Digital-first banking approach
- Byte Bank Agent (Port 10006) - Technology-focused lending

**Consumer Agent:**
- Web-based interface with real-time offer comparison
- Automatic bank agent discovery and connection management
- Session-based conversation tracking

### 7.3. Data Flow and Message Handling

**Request Processing Pipeline:**
1. User submits credit application via web interface
2. Consumer Agent validates input and constructs WFAP Intent Packet
3. Intent Packet broadcast to all discovered Bank Agents
4. Each Bank Agent processes request through risk assessment pipeline
5. Bank Agents generate and return signed WFAP Offer Packets
6. Consumer Agent aggregates responses and presents comparison
7. User reviews offers and initiates negotiation if desired
8. Final offer selection and acceptance confirmation

**Error Handling and Resilience:**
- Configurable timeouts for agent responses (300 seconds per agent)
- Graceful degradation for partial bank agent failures
- Retry mechanisms for transient network issues
- Comprehensive logging and audit trails

---

## 8. Security and Compliance

### 8.1. Digital Signature Implementation

**Message Integrity:**
- SHA-256 cryptographic hashing for message signatures
- Public/private key pairs for each agent
- Signature verification on all incoming messages

**Authentication Framework:**
- Agent identity tokens for bank identification
- Regulatory license verification
- Public key distribution via agent card endpoints

### 8.2. Regulatory Compliance

**Industry Standards:**
- NAICS industry code validation
- Jurisdiction-specific compliance checks
- Anti-money laundering (AML) screening capabilities
- Sanctions list verification

**Data Privacy and Protection:**
- Explicit data sharing consent requirements
- Secure transmission of financial data
- Audit trail maintenance for regulatory reporting

### 8.3. ESG Compliance and Reporting

**Environmental Standards:**
- Carbon emissions tracking and benchmarking
- Industry-specific environmental impact assessment
- Integration with external ESG reporting platforms

**Social and Governance Factors:**
- B-Corp and social impact certification recognition
- Corporate governance scoring
- Stakeholder impact analysis

---

## 9. Scalability and Performance Considerations

### 9.1. Horizontal Scaling

**Bank Agent Scaling:**
- Stateless agent design for easy replication
- Load balancing across multiple agent instances
- Database-backed session management for persistence

**Consumer Agent Scaling:**
- Web interface can be deployed across multiple instances
- Session affinity for user experience consistency
- Cached agent discovery for improved performance

### 9.2. Performance Optimization

**Async Processing:**
- Non-blocking I/O for all network operations
- Parallel request processing to multiple bank agents
- Streaming responses for real-time user feedback

**Caching Strategies:**
- Agent card caching to reduce discovery overhead
- ESG report caching to minimize external API calls
- Risk assessment result caching for similar profiles

### 9.3. Monitoring and Observability

**Metrics Collection:**
- Request/response latency tracking
- Success/failure rate monitoring
- ESG scoring distribution analysis
- Interest rate trend analysis

**Logging and Audit:**
- Comprehensive request/response logging
- Digital signature verification audit trails
- Regulatory compliance event logging
- Performance metrics collection

---

## 10. Future Enhancements and Roadmap

### 10.1. Protocol Evolution

**WFAP v2.0 Considerations:**
- Enhanced ESG reporting standards
- Multi-currency support
- Real-time interest rate adjustments
- Advanced risk assessment models

### 10.2. AI and Machine Learning Enhancements

**Advanced Analytics:**
- Predictive risk modeling using historical data
- Dynamic ESG scoring based on real-time data feeds
- Automated negotiation strategies
- Market trend analysis and rate optimization

### 10.3. Integration Opportunities

**External System Integration:**
- Credit bureau API integration
- Real-time ESG data feeds
- Regulatory reporting automation
- Blockchain-based transaction verification

**Ecosystem Expansion:**
- Insurance product integration
- Investment advisory services
- Supply chain financing capabilities
- International banking partnerships

---

## Conclusion

The A2A Consumer Banking System represents a sophisticated implementation of modern financial technology principles, combining AI-driven decision making with ESG considerations and regulatory compliance. The architecture demonstrates how distributed agent systems can automate complex financial processes while maintaining security, transparency, and accountability.

The system's modular design enables easy extension and customization for different financial products and regulatory environments. The integration of ESG factors into lending decisions positions the system at the forefront of sustainable finance initiatives, while the use of AI agents ensures scalable and consistent processing of credit applications.

This architecture serves as a foundation for next-generation financial services platforms that prioritize both profitability and sustainability in their lending decisions.
