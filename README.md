# IT Triaging Agent (Reinforcement Learning MVP)

## Description
This project demonstrates a lightweight IT support ticket routing agent using Q-learning and sentence embeddings.  The agent reads the content of a ticket and predicts the most suitable support team based on historical patterns and the queue length to optimize the resolution rates.  

## Features
- Embedding-based representation of ticket text (using `sentence-transformers`)
- Q-network (PyTorch) for decision-making
- Simple Q-learning agent for assigning tickets
- Mock ticket simulation in a Jupyter notebook
- Lightweight and reproducible without large datasets

## Demo / Usage
1. Open the demo notebook: [`notebooks/demo_ticket_agent.ipynb`](notebooks/demo_ticket_agent.ipynb)  
2. Run the notebook to see  ticket assignments for sample tickets.  

Example output:
Ticket: 'Email not working' -> Assigned Team: 0 | Queue: [1 0 0]
Ticket: 'VPN connection issue' -> Assigned Team: 1 | Queue: [1 1 0]
