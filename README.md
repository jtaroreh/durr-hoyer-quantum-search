# Dürr–Høyer Quantum Search

A reference implementation of the Dürr–Høyer quantum minimum-finding algorithm in Qiskit.

Given a quadratic function $f(i) = (a \cdot i^2 + b \cdot i + c) \bmod 2^m$ over $N = 2^n$ indices and a target value $t$, the algorithm finds the index minimizing the distance $|f(i) - t|$ using Grover search wrapped in the Dürr–Høyer adaptive threshold loop.

The primary goal of this repository is to demonstrate why **computed arithmetic oracles** are necessary to preserve Grover's speedup: lookup tables (QROM) require $\Omega(N)$ gates per call, which destroys the $\mathcal{O}(\sqrt{N})$ gate advantage. We also examine how this comparison inverts in fault-tolerant Clifford+T architectures and evaluate performance against classical baselines.

---

## Why Computed Oracles Matter

In textbook presentations of Grover search, the oracle is often treated as an abstract black box or an explicit lookup table (QROM). But loading an arbitrary table of $N$ values into quantum states requires $\Omega(N)$ gates per query. Running $\mathcal{O}(\sqrt{N})$ Grover iterations with a tabular oracle therefore costs $\mathcal{O}(N \sqrt{N})$ total gates—slower than a simple $\mathcal{O}(N)$ classical scan.

To retain an asymptotic advantage at the physical gate level ($\mathcal{O}(\text{poly}(\log N) \cdot \sqrt{N})$ gates total), the oracle must compute $f(i)$ coherently on the fly in $\mathcal{O}(\text{poly}(\log N))$ gates. In this project, we build a reversible arithmetic oracle using QFT phase arithmetic (`QuadraticForm`, Draper adder, and `IntegerComparator`).

---

## Quick Start

Requires Python 3.10, 3.11, or 3.12. `.[all]` pulls in the simulator (`qiskit-aer`), plotting, and test extras. A bare `pip install -e .` cannot run `demo.py`.

```bash
git clone https://github.com/jtaroreh/durr-hoyer-quantum-search.git
cd durr-hoyer-quantum-search

# Recommended: locked environment with all dependencies (uv)
uv sync --locked --all-extras

# Or install with pip in editable mode (quotes keep zsh from globbing [all])
pip install -e '.[all]'

# Run the end-to-end search demo (N=8) and save plots
python demo.py --save-plots figures/

# Run gate scaling benchmarks across NISQ and FTQC regimes
python scaling.py --max-n 12 --save-plots figures/

# Same commands after a package install (durr-heyer-* is an alias)
durr-hoyer-demo --save-plots figures/
durr-hoyer-scale --max-n 12 --save-plots figures/

# Run the test suite
pytest tests/
```

---

## How It Works

### 1. The Computed Oracle Pipeline

![Computed Quantum Distance Oracle Pipeline](figures/oracle_pipeline.png)
*Figure 1:* Pipeline for the computed distance oracle showing register allocation across 5 quantum registers, arithmetic stages, comparator, phase kick, and uncomputation.

The distance oracle maps $|i\rangle \to (-1)^{[|f(i)-t| < \text{threshold}]} |i\rangle$ in $\mathcal{O}(\text{poly}(n, m))$ gates using $n + 2m + 1$ qubits (for $m > 1$):

1. **Value Evaluation:** `QuadraticForm` evaluates $|i\rangle|0\rangle \to |i\rangle|f(i)\rangle$ via QFT phase arithmetic in $\mathcal{O}(n^2 m)$ gates.
2. **Target Subtraction:** A Draper QFT constant adder computes $f(i) - t$ in $(m+1)$-bit two's complement on the `(val, sign)` register.
3. **Absolute Value:** Controlled on the sign bit, two's complement negation yields $|f(i) - t|$.
4. **Comparison & Phase Kick:** `IntegerComparator` flags whether $|f(i) - t| < \text{threshold}$, and a `Z` gate applies the phase flip.
5. **Uncomputation:** The arithmetic steps are reversed in exact reverse order to return all scratch registers to $|0\rangle$.

### 2. The Quadratic Case vs. General Functions

Quadratic polynomials $f(i) = (a i^2 + b i + c) \bmod 2^m$ represent a uniquely favorable case for coherent arithmetic:

* **Carry-free phase coupling:** Expanding the index bits $i = \sum_{j=0}^{n-1} 2^j x_j$ shows that $i^2 = \sum_{j,k} 2^{j+k} x_j x_k$. Because quadratic terms depend only on pairwise bit products $x_j x_k$, they map directly to 2-qubit controlled-phase rotations in QFT space without carry propagation or Toffoli multiplier trees.
* **Higher-degree and non-polynomial functions:** Cubics ($i^3$), divisions ($1/x$), or cryptographic primitives (SHA-256, AES) require full reversible carry-propagation adders and Wallace-tree multipliers with $\mathcal{O}(m^2)$ Toffoli gates and additional ancilla qubits.
* **Why QROM is common in practice:** For applications lacking clean algebraic structure (such as loading arbitrary molecular Hamiltonian coefficients in quantum chemistry), optimized QROM with unary iteration remains standard despite the $\mathcal{O}(N)$ gate scaling.

### 3. The Dürr–Høyer Search Loop

Because the number of marked items is unknown and changes as the threshold decreases, the algorithm uses the Boyer–Brassard–Høyer–Tapp (BBHT) adaptive schedule:

1. Sample an initial random index $i_0$ and set $\text{threshold} = |f(i_0) - t|$.
2. Draw Grover iteration count $j$ uniformly from $\{0,\ldots,\lceil m_{\text{BBHT}}\rceil-1\}$. After a non-improving round, grow $m_{\text{BBHT}} \leftarrow \min((6/5) m_{\text{BBHT}}, \sqrt{N})$. After an improvement, reset $m_{\text{BBHT}}$ to $1$.
3. Measure an index $i'$ and verify classically. If $|f(i') - t| < \text{threshold}$, update the threshold.
4. Stop when distance 0 is found or the simulator query ceiling $\lceil 15\sqrt{N}\rceil + 10$ is reached. The paper's expected query count is $(45/4)\sqrt{N} + 0.7\log_2^2 N \approx 11.25\sqrt{N}$; the tables use that bound, not the simulator ceiling.

![Dürr–Høyer Search Trajectory](figures/durr_hoyer_trajectory.png)
*Figure 2:* Left: Search space landscape $|f(i)-t|$ with global minimum marked. Right: Dynamic threshold ladder stepping down round-by-round.

![Grover Amplitude Amplification](figures/grover_amplitudes.png)
*Figure 3:* Measurement probabilities after threshold discovery, comparing amplified probability on the closest set (>94%) against the uniform state ($1/N = 12.5\%$).

---

## Scaling & Complexity Analysis

### White-Box vs. Black-Box and Classical Baselines

When analyzing quantum search speedups, access assumptions matter:

1. **Black-Box Model:** Assumes an opaque oracle where classical search requires exhaustive scanning over all $N = 2^n$ candidates.
2. **White-Box Model:** When $f(i) = (a i^2 + b i + c) \bmod 2^m$ is explicitly known (necessary to build the circuit in $\mathcal{O}(\text{poly}(\log N))$ gates), classical solvers can exploit structure:
   * **Periodicity:** $f(i + 2^m) \equiv f(i) \pmod{2^m}$. For $N > 2^m$, evaluating $2^m$ candidates covers all unique values, bounding classical checks at $\min(N, 2^m)$.
   * **Algebraic Solvers:** Exact congruences $f(i) \equiv t \pmod{2^m}$ can be solved directly in $\mathcal{O}(\text{poly}(m))$ using 2-adic / Hensel lifting.

We implement both classical baselines (`classical_closest` for black-box and `classical_structured_closest` for period reduction) in [closest_search/search.py](closest_search/search.py).

### Regime 1: Unstructured Index Space ($m = n + 1, N \le 2^m$)

Here we benchmark scaling for $f(i) = (2i^2 + 3i + 1) \bmod 2^{n+1}$ with target $t=6$ and threshold $1$ ($k=1$ unique marked item).

![NISQ Gate Complexity Scaling](figures/nisq_scaling.png)
*Figure 4:* Scaling of transpiled NISQ gates (basis $\{u, cx\}$) for Single-Run Grover and Dürr–Høyer search vs. classical bit operations and CPU instructions.

**Note on units and empirical slopes:**
Comparing quantum gates to classical instructions illustrates algorithmic complexity, not wall-clock speedup (a 2-qubit gate cycle is much slower than a CPU instruction). While Grover achieves $\mathcal{O}(\sqrt{N})$ query complexity, total gate complexity is $\mathcal{O}(\sqrt{N} \log^3 N)$. At $n \le 12$, the polylogarithmic growth of the arithmetic oracle makes the empirical quantum gate slope $\approx 1.0$ (close to the CPU instruction slope), but it clearly outperforms classical bit-level operations (slope $\approx 1.43$).

<details>
<summary>Gate scaling data table (n = 2 to 12)</summary>

| $n$ | $m$ | $N=2^n$ | Q-Oracle (CNOTs) | Q-Iter Gates | Single-Run Q-Gates (Optimal $R$) | Dürr–Høyer Q-Gates (11.25√N + 0.7log²N) | C-BlackBox Bit-Ops | C-BlackBox CPU-Ops |
| --: | --: | ------: | ----------------: | -----------: | --------------------------------: | ---------------------------------------: | -----------------: | ------------------: |
| 2 | 3 | 4 | 515 (246) | 520 | 520 | 13,156 | 52 | 12 |
| 3 | 4 | 8 | 964 (452) | 982 | 1,964 | 37,434 | 200 | 24 |
| 4 | 5 | 16 | 1,547 (734) | 1,580 | 4,740 | 88,796 | 656 | 48 |
| 5 | 6 | 32 | 2,284 (1,096) | 2,367 | 9,468 | 192,057 | 1,952 | 96 |
| 6 | 7 | 64 | 3,235 (1,578) | 3,420 | 20,520 | 393,984 | 5,440 | 192 |
| 7 | 8 | 128 | 4,374 (2,160) | 4,660 | 37,280 | 752,959 | 14,464 | 384 |
| 8 | 9 | 256 | 5,777 (2,898) | 6,184 | 74,208 | 1,390,163 | 37,120 | 768 |
| 9 | 10 | 512 | 7,402 (3,756) | 7,956 | 135,252 | 2,476,372 | 92,672 | 1,536 |
| 10 | 11 | 1,024 | 9,341 (4,806) | 10,069 | 251,725 | 4,329,670 | 226,304 | 3,072 |
| 11 | 12 | 2,048 | 11,536 (5,996) | 12,533 | 438,655 | 7,442,307 | 542,720 | 6,144 |
| 12 | 13 | 4,096 | 14,095 (7,414) | 15,330 | 766,500 | 12,582,864 | 1,282,048 | 12,288 |

*Table notes:*
* **Empirical slopes** ($n \le 12$): Single-Run Grover $\sim N^{1.00}$; Dürr–Høyer $\sim N^{0.96}$; Classical bit-ops $\sim N \log^2 N$; Classical CPU-ops $\sim 3N$.
* **Single-Run Grover:** Uses exact optimal rotation count $R = \arg\max_R \sin^2((2R+1)\arcsin\sqrt{k/N})$.
* **Dürr–Høyer Bound:** Evaluated at expected query count $(45/4)\sqrt{N} + 0.7\log_2^2 N$.
* **Classical CPU-Ops:** Evaluates $(a \cdot i^2 + b \cdot i + c) \bmod 2^m$ in $\approx 3$ word instructions.
* **Classical Bit-Ops:** Standard software bit-level gate model: $n^2 + nm + m$ operations.

</details>

### Regime 2: Periodic Index Space (Fixed $m = 4, n > m$)

When $N > 2^m$, periodicity $f(i + 2^m) \equiv f(i) \pmod{2^m}$ caps classical evaluations at $2^m = 16$.

<details>
<summary>Periodic scaling data table (m = 4, n = 4 to 8)</summary>

| $n$ | $m$ | $N=2^n$ | $2^m$ | C-BlackBox Evals | C-Struct Evals | C-BlackBox Ops | C-Struct Ops |
| --: | --: | ------: | ----: | ---------------: | -------------: | -------------: | -----------: |
| 4 | 4 | 16 | 16 | 16 | 16 | 576 | 576 |
| 5 | 4 | 32 | 16 | 32 | 16 | 1,568 | 784 |
| 6 | 4 | 64 | 16 | 64 | 16 | 4,096 | 1,024 |
| 7 | 4 | 128 | 16 | 128 | 16 | 10,368 | 1,296 |
| 8 | 4 | 256 | 16 | 256 | 16 | 25,600 | 1,600 |

</details>

### Empirical QROM vs. Computed Arithmetic ($n \le 6$)

We compare an explicit value-loading QROM oracle (optimized with Gray-code multi-controlled $X$ transitions) against the coherent `QuadraticForm` oracle, transpiled to basis $\{u, cx\}$ with `optimization_level=1`:

![Empirical QROM vs Computed Oracle](figures/qrom_vs_coherent_nisq.png)
*Figure 5:* Left: Transpiled gate counts for Gray-code QROM vs. coherent arithmetic. Right: QROM gate overhead multiplier ($11.38\times$ at $N=64$).

While coherent QFT arithmetic scales as $\mathcal{O}(n^2 m)$, tabular QROM scales as $\Theta(m \cdot 2^n)$, requiring **1.56× more gates** at $n=3$ ($N=8$), **3.09×** at $n=4$ ($N=16$), and **11.38×** at $n=6$ ($N=64$).

<details>
<summary>QROM vs Computed gate count table (n = 2 to 6)</summary>

| $n$ | $m$ | $N=2^n$ | QROM Oracle Gates | Computed Oracle Gates | Ratio (QROM / Computed) |
| --: | --: | ------: | -----------------: | --------------------: | -----------------------: |
| 2 | 3 | 4 | 516 | 515 | 1.00x |
| 3 | 4 | 8 | 1,501 | 964 | 1.56x |
| 4 | 5 | 16 | 4,777 | 1,547 | 3.09x |
| 5 | 6 | 32 | 13,404 | 2,284 | 5.87x |
| 6 | 7 | 64 | 36,819 | 3,235 | 11.38x |

</details>

---

## Fault-Tolerant (Clifford+T) Scaling & Crossover

In fault-tolerant quantum computing (FTQC), arbitrary continuous rotations (such as QFT phase angles) must be synthesized via magic state distillation factories (Ross & Selinger 2016). Discrete operations like Toffolis in QROM lookup trees decompose directly into discrete Clifford+T gates without rotation synthesis error ($\varepsilon = 0$).

### 1. Pipeline Breakdown

| Stage | Circuit | Compilation Model |
| :--- | :--- | :--- |
| **Value Loader (Coherent)** | `QuadraticForm` | QFT phase coupling: $K_{\text{total}} = 12nm + 7n(n-1)m + 6m(m-1)$ rotations synthesized to precision $\varepsilon / K_{\text{total}}$. |
| **Value Loader (QROM)** | Unary Iteration (Babbush 2018) | Binary selection tree: $N-1$ Toffolis ($4T$ each with measurement assistance). Reversible uncomputation costs $8(N-1)$ $T$-gates with $\varepsilon = 0$. |
| **Downstream Pipeline** | Subtraction, Absolute Value, Comparator | Draper adder + absolute value with $K_{\text{downstream}} = 12m^2+8m+2$ rotations plus ripple-carry comparator with $16(m-1)$ $T$-gates. |

<details>
<summary>Mathematical details: rotation decomposition & Clifford+T synthesis</summary>

1. **Rotation Synthesis (Ross & Selinger 2016):** An arbitrary single-qubit Z-rotation $R_z(\theta)$ synthesized to precision $\varepsilon_{\text{rot}} = \varepsilon / K$ requires approximately:

   $$T(R_z, \varepsilon_{\text{rot}}) \approx \left\lceil 3 \log_2\left(\frac{K}{\varepsilon}\right) \right\rceil$$

2. **Coherent Arithmetic Loader:** Decomposes into $nm$ controlled-phase gates for linear terms, $nm$ for diagonal terms, $\frac{1}{2}n(n-1)m$ doubly-controlled phase gates for off-diagonal terms, and $m(m-1)$ rotations for the register QFT. With compute and uncompute ($2\times$):

   $$K_{\text{total}} = 12nm + 7n(n-1)m + 6m(m-1)$$

   $$T_{\text{coherent}}(n, m, \varepsilon) = K_{\text{total}} \cdot \left\lceil 3 \log_2\left(\frac{K_{\text{total}}}{\varepsilon}\right) \right\rceil$$

3. **QROM Unary Iteration Loader (Babbush et al. 2018):**

   $$T_{\text{QROM}}(n) = 8(2^n - 1)$$

</details>

### 2. Continuous QFT vs. Discrete QROM Crossover

![Fault-Tolerant Clifford+T Crossover](figures/ftqc_crossover.png)
*Figure 6:* Clifford+T gate cost for optimal discrete QROM ($8(N-1)$ T-gates) vs. continuous-phase QuadraticForm arithmetic across synthesis precision budgets $\varepsilon$, showing the crossover around $N \approx 10^6$.

<details>
<summary>Fault-tolerant value loader comparison table (T-count, n = 2 to 20)</summary>

`python scaling.py --markdown` prints every regime table. `--max-n` only truncates the empirical NISQ and QROM tables. The FTQC loader and full-oracle rows use a fixed $n$ grid through $20$. The value-loader table ($T_{\text{load}}$) is:

| $n$ | $m$ | $N=2^n$ | QROM Loader $T$ ($8(N-1)$) | Coherent Loader $T$ ($\varepsilon=10^{-4}$) | Coherent Loader $T$ ($\varepsilon=10^{-6}$) | Coherent Loader $T$ ($\varepsilon=10^{-8}$) | Coherent Loader $T$ ($\varepsilon=10^{-10}$) | Loader Ratio ($\varepsilon=10^{-6}$) |
| --: | --: | ------: | -------------------------: | -------------------------------------------: | -------------------------------------------: | -------------------------------------------: | --------------------------------------------: | -------------------------------------: |
| 2 | 3 | 4 | 24 | 9,300 | 12,300 | 15,300 | 18,300 | QROM 512.5x cheaper |
| 3 | 4 | 8 | 56 | 25,344 | 33,024 | 40,704 | 48,384 | QROM 589.7x cheaper |
| 4 | 5 | 16 | 120 | 53,820 | 69,420 | 85,020 | 100,620 | QROM 578.5x cheaper |
| 6 | 7 | 64 | 504 | 164,724 | 209,244 | 253,764 | 298,284 | QROM 415.2x cheaper |
| 8 | 9 | 256 | 2,040 | 371,448 | 467,928 | 564,408 | 660,888 | QROM 229.4x cheaper |
| 10 | 11 | 1,024 | 8,184 | 712,800 | 891,000 | 1,069,200 | 1,247,400 | QROM 108.9x cheaper |
| 12 | 13 | 4,096 | 32,760 | 1,215,240 | 1,511,640 | 1,808,040 | 2,104,440 | QROM 46.1x cheaper |
| 14 | 15 | 16,384 | 131,064 | 1,922,760 | 2,380,560 | 2,838,360 | 3,296,160 | QROM 18.2x cheaper |
| 16 | 17 | 65,536 | 524,280 | 2,843,760 | 3,512,880 | 4,182,000 | 4,851,120 | QROM 6.7x cheaper |
| 18 | 19 | 262,144 | 2,097,144 | 4,076,298 | 5,013,378 | 5,950,458 | 6,887,538 | QROM 2.4x cheaper |
| 19 | 20 | 524,288 | 4,194,296 | 4,815,360 | 5,909,760 | 6,949,440 | 8,043,840 | QROM 1.4x cheaper |
| 20 | 21 | 1,048,576 | 8,388,600 | 5,580,960 | 6,849,360 | 8,117,760 | 9,386,160 | Coh 1.22x cheaper |

</details>

<details>
<summary>Fault-tolerant full distance oracle comparison table (T-count, n = 2 to 20)</summary>

Clifford+T cost for the complete distance oracle (loader + subtraction + absolute value + comparator + phase kick + uncomputation):

| $n$ | $m$ | $N=2^n$ | QROM Full Oracle $T$ ($\varepsilon=10^{-6}$) | Coh Full Oracle $T$ ($\varepsilon=10^{-4}$) | Coh Full Oracle $T$ ($\varepsilon=10^{-6}$) | Coh Full Oracle $T$ ($\varepsilon=10^{-8}$) | Coh Full Oracle $T$ ($\varepsilon=10^{-10}$) | Oracle Ratio ($\varepsilon=10^{-6}$) |
| --: | --: | ------: | -------------------------------------------: | ------------------------------------------: | ------------------------------------------: | ------------------------------------------: | -------------------------------------------: | -------------------------------------: |
| 2 | 3 | 4 | 11,312 | 18,492 | 24,038 | 29,718 | 35,398 | QROM 2.1x cheaper |
| 3 | 4 | 8 | 19,766 | 41,686 | 53,886 | 66,086 | 78,286 | QROM 2.7x cheaper |
| 4 | 5 | 16 | 30,622 | 79,822 | 102,262 | 124,360 | 146,800 | QROM 3.3x cheaper |
| 6 | 7 | 64 | 59,386 | 217,364 | 274,804 | 332,244 | 389,684 | QROM 4.6x cheaper |
| 8 | 9 | 256 | 99,446 | 462,406 | 579,806 | 697,206 | 814,606 | QROM 5.8x cheaper |
| 10 | 11 | 1,024 | 154,834 | 855,340 | 1,064,380 | 1,273,420 | 1,482,460 | QROM 6.9x cheaper |
| 12 | 13 | 4,096 | 237,816 | 1,424,210 | 1,761,156 | 2,100,236 | 2,439,316 | QROM 7.4x cheaper |
| 14 | 15 | 16,384 | 407,844 | 2,211,770 | 2,726,010 | 3,240,250 | 3,754,490 | QROM 6.7x cheaper |
| 16 | 17 | 65,536 | 881,530 | 3,229,258 | 3,970,498 | 4,711,738 | 5,452,978 | QROM 4.5x cheaper |
| 18 | 19 | 262,144 | 2,546,032 | 4,576,028 | 5,602,828 | 6,629,628 | 7,656,428 | QROM 2.2x cheaper |
| 19 | 20 | 524,288 | 4,690,800 | 5,376,784 | 6,570,424 | 7,709,344 | 8,902,984 | QROM 1.4x cheaper |
| 20 | 21 | 1,048,576 | 8,940,582 | 6,213,962 | 7,591,602 | 8,963,780 | 10,341,420 | Coh 1.18x cheaper |

</details>

### 3. Practical Notes on Fault-Tolerant Compilation

* **NISQ vs. FTQC Scaling:** In NISQ gate counts, coherent arithmetic looks consistently cheaper because continuous phase rotations are treated as single unit gates. In fault-tolerant regimes, each continuous rotation requires expensive magic state distillation ($\approx 60\text{--}150$ $T$-gates), whereas discrete Toffolis require only $4T$ gates. This makes QROM significantly cheaper for $N \le 10^5$.
* **Crossover Sensitivity:** The $N \approx 10^6$ crossover in Figure 6 compares theoretical best-case unary iteration QROM ($8(N-1)$ $T$-gates) against unmerged continuous QFT rotation synthesis. Compiling discrete reversible arithmetic (e.g., Toffoli-based adders with $\mathcal{O}(n^2)$ $T$-gates) shifts the crossover down near $N \approx 10^3$.
* **Conservative Gridsynth Proxy:** Treating all rotations in `QuadraticForm` as arbitrary angles provides an upper bound. Merging diagonal terms ($x_j^2 = x_j$) and compiling exact dyadic angles ($CZ, CS, T$) directly further reduces coherent $T$-counts.

---

## Codebase Overview

| File | Description |
| :--- | :--- |
| `closest_search/circuits.py` | Quantum circuit builders for arithmetic and QROM oracles (`value_function`, `distance_oracle`, `diffuser`) |
| `closest_search/ftqc.py` | Analytical Clifford+T cost models and resource proxies |
| `closest_search/nisq.py` | NISQ complexity accounting, exact Grover rotation formulas, and scaling records |
| `closest_search/search.py` | Dürr–Høyer search driver, BBHT schedule, and classical baselines |
| `closest_search/plotting.py` | Matplotlib & Seaborn visualization routines |
| `demo.py` | CLI demo for search progression, measurement distributions, and figure export |
| `scaling.py` | Benchmark script for gate scaling, QROM comparison, and Clifford+T crossover analysis |
| `tests/` | Unit tests for circuits, baseline solvers, FTQC models, plotting, and CLI failure modes |
