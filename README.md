# CLT in Action: Understanding the Central Limit Theorem Through Finance and Communications

The **Central Limit Theorem (CLT)** is one of the most powerful and universal results in probability theory. This repository explores the CLT from first principles, builds intuition through simulations, and demonstrates its practical usage in two major domains:

- **Finance** — portfolio returns, risk modeling, Monte Carlo simulations  
- **Communications (with a focus on aircraft antenna systems)** — noise modeling, signal aggregation, and interference analysis  

The goal is to make the CLT *intuitive*, *visual*, and *applied*.

---

## 🎯 Project Goals

- Build a deep understanding of the Central Limit Theorem  
- Show how CLT emerges naturally in real-world systems  
- Provide clean, modular Python code for simulations  
- Demonstrate CLT applications in:
  - Finance (portfolio returns, risk modeling, Monte Carlo)
  - Communications (noise aggregation, interference)
  - Antenna systems (Gaussian noise emergence)
- The project is fully Python-based — no Jupyter notebooks — to support reproducibility, maintainability, and integration into larger codebases.
- Create reproducible notebooks and visualizations  
- Serve as an educational reference for students, engineers, and researchers  

---

## 📂 Repository Structure

```
clt-in-action/
│
├── clt/
│   ├── __init__.py
│   ├── core.py                 
│   ├── distributions.py        
│   ├── finance.py             
│   ├── communications.py       
│   └── plotting.py             
│
├── examples/
│   ├── run_basic_clt.py
│   ├── run_finance_demo.py
│   ├── run_communications_demo.py
│   └── run_aircraft_antenna_noise.py
│
├── docs/
│   ├── theory.md
│   ├── finance.md
│   ├── communications.md
│   └── antennas.md
│
├── plots/                      
├── README.md
└── requirements.txt
```

---

## 📘 What is the Central Limit Theorem?

The **Central Limit Theorem** states that the sum (or average) of many independent random variables tends toward a **normal distribution**, regardless of the original distribution, provided certain conditions are met.

This explains why Gaussian models appear everywhere — even when the underlying physics or economics are not Gaussian at all.

---

## 📈 Applications Covered

### **1. Finance**
- Portfolio return aggregation  
- Risk estimation  
- Monte Carlo simulations  
- Log-return modeling  
- Why Gaussian assumptions sometimes work — and when they fail  

### **2. Communications**
- Additive White Gaussian Noise (AWGN)  
- Interference from many independent sources  
- Signal processing in antenna arrays  
- Why aircraft antenna noise is modeled as Gaussian  
- CLT in OFDM and multi-carrier systems  

---

## 🛫 Aircraft Antenna Case Study

Aircraft communication systems experience noise from many independent sources:
- thermal noise  
- atmospheric noise  
- interference from other onboard electronics  
- multipath reflections  

The CLT explains why the **aggregate noise** is well-modeled as **Gaussian**, even though individual sources are not.

This repo includes:
- simulations of aggregated noise  
- visualizations of convergence to Gaussian  
- antenna-specific signal models  

---

## 🧪 How to Run the Code

```bash
git clone https://github.com/AakashVats15/clt-in-action
cd clt-in-action
pip install -r requirements.txt
```

Then explore the code:
```
python examples/run_basic_clt.py
python examples/run_finance_demo.py
python examples/run_communications_demo.py
python examples/run_aircraft_antenna_noise.py
```

---

## 🤝 Contributions

Contributions, suggestions, and improvements are welcome.  
Feel free to open issues or submit pull requests.


---

## ⭐ If you find this useful…

Please consider starring the repository to support the project.
