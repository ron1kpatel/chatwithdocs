# Chat with Documents CLI Tool

## Overview

The Chat with Documents CLI Tool is a command-line interface application designed to analyze PDF documents and provide detailed answers to user queries based on the content of those documents. It leverages advanced language models and embeddings to perform context-aware retrieval and response generation.

## Features

- Load and process multiple PDF documents.
- Split documents into manageable chunks for efficient processing.
- Generate embeddings using Google Generative AI Embeddings.
- Retrieve relevant document sections based on user queries.
- Provide detailed and contextually accurate answers.
- Display response time for performance insights.

## Installation and Setup

### Prerequisites

Before setting up the project, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/chat-with-docs-cli.git
cd chat-with-docs-cli
```

### Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Environment Variables

The tool requires API keys for Groq and Google Generative AI Embeddings. Create a `.env` file in the root directory of the project and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Usage

To use the tool, run the following command in your terminal:

```bash
python chat_with_docs.py --docs path/to/document1.pdf path/to/document2.pdf --query "Your question here"
```

Replace `path/to/document1.pdf`, `path/to/document2.pdf`, and `Your question here` with your actual file paths and query.

### Example

```bash
python chat_with_docs.py --docs docs/sample1.pdf docs/sample2.pdf --query "What is the main topic discussed in these documents?"
```


