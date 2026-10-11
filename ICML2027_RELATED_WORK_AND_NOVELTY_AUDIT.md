# ICML 2027 — Novelty Audit
## Knowledge organization → selective specialization/compression

**Stan literatury sprawdzony: 9 października 2026**
(wersja 2: audyt GPT + weryfikacja źródeł i uzupełnienie przez Claude)

**Zakres:** ten dokument dotyczy wyłącznie paperu ICML 2027. Projekt UCT (języki) może korzystać z tych samych narzędzi i częściowo z tych samych celów (języki jako targety). Projekt Waikato (streaming) jest osobnym projektem i nie jest częścią tego paperu.

---

## 0. Co się zmieniło w wersji 2

1. **Weryfikacja źródeł.** Sprawdzono najnowsze i najbardziej ryzykowne pozycje (#16, #19, #20, #21, #26, #27, #28, #29). Wszystkie istnieją i są opisane trafnie.
2. **Dodano 12 brakujących prac** (#30–#41). Najważniejsza luka w wersji 1: brak linii *„Does localization inform …?”*, która jest jednocześnie głównym ryzykiem i najlepszym framingiem.
3. **Obecna teza repo jest już opublikowana.** *Decodable But Not Detachable* (08.2026) pokazuje w skali 3 rodzin × 8 domen to, co repo pokazuje na TinyLlamie. Patrz §4F i §12.
4. **Dodano wymagania metodologiczne** (§7a): cyrkularność, predyktory bazowe, liczba targetów, held-out targets, niezależne prunery.
5. **Przeprojektowano pilot** (§13) i dodano harmonogram z decyzją go/no-go (§14).

---

## 1. Zamrożony problem badawczy

Interesuje nas nie „kolejny pruning score”, lecz pytanie:

> **Jak wiedza i kompetencje są zorganizowane w ogólnym LLM oraz czy ta organizacja wyjaśnia, jak łatwo można wydzielić z niego małego specjalistę zachowującego wybrane kompetencje?**

Przykład:

\[
M_{\mathrm{general}}
\rightarrow
M_{\mathrm{medicine}}
\]

przy zachowaniu medycyny oraz minimalnego „core” językowo-rozumującego, bez konieczności zachowania wiedzy o rolnictwie, astronomii itd.

---

# 2. Najważniejszy wynik audytu

Cztery proste wersje problemu są już zajęte:

1. **„Zróbmy domain/task-specific experta przez pruning”** — istnieje wiele prac.
2. **„Ile parametrów naprawdę potrzebuje konkretne zadanie?”** — LLM-Sieve pyta o to wprost.
3. **„Gdzie są zlokalizowane capabilities/domains?”** — mocno rozwinięta literatura 2024–2026.
4. **„Domeny są dekodowalne, ale nie dają się wyciąć na poziomie neuronów”** — *Decodable But Not Detachable* (08.2026). To jest dokładnie obecna teza repo.

Znane jest też, że **kompresja degraduje kompetencje nierówno** (fact recall vs ICL; wiedza head vs tail). Nie ma natomiast mechanistycznego wyjaśnienia, *dlaczego*.

Najbardziej obiecująca luka, której nie znalazłem zamkniętej przez istniejące prace:

> **Czy funkcjonalna/causal modularity w gęstym modelu wyjaśnia i przewiduje specialist compressibility/extractability — i na jakiej granularności strukturalnej?**

\[
\boxed{
\text{functional organization at scale }g
\rightarrow
\text{specialist compression frontier}
}
\]

gdzie \(g\) może oznaczać neurony, grupy kanałów MLP, attention heads, warstwy, pathways lub low-dimensional subspaces.

Najostrzejszy framing to: **„Does Localization Inform Compression?”**. Jest to naturalna kontynuacja linii Hase i in. (editing) oraz Lee i in. (unlearning), w których odpowiedź okazała się w dużej mierze negatywna.

---

# 3. Audyt najbliższych prac

Legenda:
- **Causal:** czy wykonuje interwencje/ablacje mające wykazać funkcjonalną konieczność.
- **Expert:** czy tworzy specjalistyczny/przycięty model.
- **Minimum:** czy explicite szuka minimalnej pojemności/submodelu dla zadania.
- **Explains differences:** czy wyjaśnia, dlaczego jedna domena/capability jest łatwiejsza do wydzielenia/kompresji od innej.
- **✓** w kolumnie „#”: pozycja zweryfikowana 9.10.2026 (istnieje, opis zgodny z abstraktem).

| # | Praca | Status | Główny wkład | Granularność | Causal | Expert | Minimum | Explains differences? | Znaczenie dla nas |
|---|---|---|---|---|---|---|---|---|---|
| 1 | SparseGPT | ICML 2023 | One-shot unstructured pruning dużych LLM | weights | Nie | Nie | Nie | Nie | Baseline ogólnego pruningu; **kandydat na niezależny pruner do frontierów** |
| 2 | LLM-Pruner | NeurIPS 2023 | Structured task-agnostic pruning + recovery | coupled structures | Częściowo | Nie | Nie | Nie | Baseline structured pruning |
| 3 | Wanda | ICLR 2024 | Weight × activation importance, retraining-free | weights | Nie | Nie | Nie | Nie | Fundamentalny baseline; **kandydat na niezależny pruner do frontierów** |
| 4 | Sheared LLaMA | ICLR 2024 | Targeted structured pruning do zadanej architektury | layers/heads/dims | Nie | Mały generalist | Zadany rozmiar | Nie | Generalist → mniejszy generalist |
| 5 | FLAP | AAAI 2024 | Hardware-friendly structured pruning + compensation | channels/modules | Nie | Nie | Nie | Nie | Baseline structured |
| 6 | SliceGPT | ICLR 2024 | Redukcja wymiaru przez rows/columns | hidden dimension | Nie | Nie | Nie | Nie | Baseline real size/speed |
| 7 | ShortGPT | Findings ACL 2025 | Layer redundancy i layer dropping | layers | Nie | Nie | Nie | Częściowo | Pokazuje znaczenie granularności |
| 8 | MoDeGPT | ICLR 2025 | Modular decomposition do kompresji | matrix modules | Nie | Nie | Nie | Nie | Ogólny compression baseline |
| 9 | Let LLM Tell What to Prune and How Much | ICML 2025 | Dynamiczna alokacja pruning ratio między strukturami | multiple structures | Częściowo | Nie | Nie | Częściowo | Bliskie pytaniu „która struktura ważna” |
| 10 | D-Pruner | Findings NAACL 2024 | Domain-specific LLM extractor; general + domain importance | weights | Nie | **Tak** | Nie | Nie | **Zajmuje prosty medical/legal expert pruning** |
| 11 | Cus-Prun | Findings ACL 2025 | Generalist → lightweight expert w wymiarze language/domain/task | neurons | Nie | **Tak** | Nie | Nie | **Bardzo blisko prostego pomysłu experta** |
| 12 | TrimLLM | ACL 2025 | Domain-specific layer dropping; medical/legal/finance | layers | Empiryczna weryfikacja | **Tak** | Nie | Częściowo | Domain specialization na poziomie warstw |
| 13 | FineScope | arXiv 2025 | SAE-guided domain data + structured pruning + distillation | SAE features + structures | Nie | **Tak** | Nie | Nie | Interpretability już używane w domain compression |
| 14 | Frustratingly Easy Task-aware Pruning | arXiv 2025 | Shared/general/task-only groups z activation norms | weights/channels | Nie | Tak | Nie | Częściowo | **Shared vs exclusive groups już istnieją** |
| 15 | PPC-GPT | EMNLP 2025 | Task-specific SLM przez pruning + distillation | weights/model | Nie | **Tak** | Nie | Nie | Task-specific compression |
| 16 ✓ | LLM-Sieve | arXiv 2025 | „How many parameters does your task really need?”; 3.8B–70B, 20–75% wag usuwalne przy 1–5% straty | per-matrix low-rank/pruning | Nie | **Tak** | **Tak** | Częściowo: bottleneck matrices | **Zajmuje prosty claim minimalnego subsetu** |
| 17 | SEAP | AAAI 2026 | Task-relevant sparse activation paths + router | neurons/pathways | Transfer tests | **Tak/dynamic** | Nie | Częściowo | **Task-specific pathways już istnieją** |
| 18 | MixCal | EACL 2026 | Domain-specific calibration poprawia pruning/quantization | calibration data | Nie | Tak | Nie | Nie | Domain data alone strongly affects compression |
| 19 ✓ | SubspacePath Pruner | ICML 2026 | Domain-basis synthesis + per-layer probes → head-wise pathway w budżecie; LLaMA-2-13B | heads/pathways/subspaces | Probe-based | **Tak** | Budżet zadany | Częściowo | **Najbliższa praca do representation→specialist pathway** |
| 20 ✓ | KOFF | arXiv 05.2026 | Sparse shared core + swappable domain memories (LoRA + KV); Llama/Qwen 3–8B | structured neurons + adapters/KV | Nie | **Tak/modular** | Nie | Częściowo | **Shared core + specialist memory już eksplorowane** |
| 21 ✓ | Capability-Guided Compression | arXiv 03.2026 | SAE capability-density → compression budgets; tylko GPT-2 Medium, wynik negatywny | heads/components | Nie | Nie | Nie | Częściowo | Słaba empirycznie, ale zajmuje framing „capability-aware budget” |
| 22 | NeuronMoE | EACL 2026 | Language-specific neurons kierują allocation experts | neurons/layers | Nie | Modular experts | Nie | **Tak dla języków** | Ważne dla projektu UCT |
| 23 | Language-Specific Neurons | ACL 2024 | Mała populacja neuronów specyficznych językowo | neurons | Steering/masking | Nie | Nie | Częściowo | Silna modularność językowa |
| 24 | Capability Localization | ICLR 2025 | Capabilities, nie pojedyncze fakty, mogą być lokalizowane | neurons | Tak/decoupling | Nie | Nie | Częściowo | Localization sama w sobie zajęta |
| 25 | Cross-Lingual Generalization and Compression | ACL 2025 | Language-specific → shared semantic neurons | neurons | Steering | Nie | Nie | **Tak** | Shared/specific representations |
| 26 ✓ | Compressed Sensing for Capability Localization | ICML 2026 | 5 głów może obniżyć math/code nawet o 60% przy małym wpływie na inne zadania; Llama/Qwen 1–14B | attention heads | **Tak** | Nie | Nie | Częściowo | Bardzo mocne capability localization **na poziomie głów** |
| 27 ✓ | Decodable But Not Detachable | arXiv 08.2026 | 3 rodziny 1.5–7B, 8 domen, 939k neuronów FFN: przedmioty dekodowalne (>85%), ale płaskie damage matrices; język/modalność prawie diagonalne; shells rosną ze skalą; zależy od granularności danych treningowych | FFN neurons | **Tak** | Nie | Nie | **Tak: training granularity** | **Kluczowy wynik i zagrożenie: zajmuje obecną tezę repo. Badają tylko neurony FFN, nie głowy/grupy** |
| 28 ✓ | Modular Cognitive Architecture Emerges | arXiv 2026 (Han, Andreas, Fedorenko, de Varda) | 46 zadań, 4 cognitive domains; nakładanie się neuronów odzwierciedla sieci w mózgu | MLP neurons/circuits | **Tak** | Nie | Nie | Częściowo | Modularność na szerszych cognitive domains |
| 29 ✓ | CRANE | arXiv 01.2026 | Causal language-specific neurons; „selective but non-exclusive” | neurons | **Tak** | Nie | Nie | Częściowo | Potwierdza causal language modularity |
| 30 | **Does Localization Inform Editing?** (Hase i in.) | NeurIPS 2023 | Lokalizacja (causal tracing) nie przewiduje, gdzie edycja działa najlepiej | layers | **Tak** | Nie | Nie | — | **Bezpośredni precedens: „localization → intervention success” często zawodzi** |
| 31 | **Does Localization Inform Unlearning?** (Lee i in.) | EMNLP 2025 | Unlearning w prawdziwym regionie daje tylko marginalny zysk vs losowy region tej samej wielkości; wiele alternatywnych regionów | value vectors | **Tak** | Nie | Nie | — | **Główne ryzyko dla hipotezy i najlepszy framing paperu** |
| 32 | The Cost of Down-Scaling LMs (Jin i in.) | arXiv 2023 | >30% redukcji szkodzi fact recall; 60–70% redukcji oszczędza ICL | model/weights | Nie | Nie | Nie | **Tak, empirycznie** | Różna kompresowalność kompetencji jest znana; nowe jest tylko *wyjaśnienie* |
| 33 | The Asymmetric Harms of LLM Compression | arXiv 08.2026 | 3 LLM × 11 metod; kompresja bardziej uderza w wiedzę head niż tail; zmiany biasu ukryte w agregatach | — | Nie | Nie | Nie | Opisowo | **Confounder: częstość wiedzy w pretrainingu** |
| 34 | Are Neural Nets Modular? (Csordás i in.) | ICLR 2021 | Funkcjonalna modularność przez differentiable weight masks | weights | **Tak** | Nie | Nie | Częściowo | Obowiązkowy precedens metodologiczny dla \(\mathcal M_d\) |
| 35 | Emergent Modularity in Pre-trained Transformers | Findings ACL 2023 | Neurony FFN tworzą emergentne moduły funkcjonalne (MoEfication) | neurons/experts | Częściowo | Nie | Nie | Częściowo | Precedens dla „dense → modules” |
| 36 | Gradient Routing | arXiv 2024 | Maskowanie gradientów lokalizuje zdolności w wybranych regionach już w treningu | regions | **Tak** | Nie | Nie | Częściowo | Modularność jako własność treningu; wspiera wniosek #27 |
| 37 | TASP (Task-Aware Spectral Pruning) | arXiv 2026 | Mixture-of-masks per task family; pilot sprawdza, czy sygnał spektralny przewiduje wrażliwość zadania (przechodzi dla Llama-3 8B/70B, nie dla Qwen2.5-1.5B) | modules | Kalibracja ablacjami | **Tak/routed** | Nie | Częściowo | Bliskie idei „przewidywania” kompresowalności, ale na poziomie modułów |
| 38 | Less is MoE (Fisher-MoE) | arXiv 06.2026 | Usunięcie 12 z ~1,35 mln wymiarów FFN niszczy GSM8K, a wiedza faktograficzna zostaje | FFN intermediate dims (MoE) | **Tak** | Tak | Nie | Częściowo | Wspiera tezę „granularność zależy od kompetencji”; konkurencja w MoE |
| 39 | Extracting Small Translation Specialists | arXiv 05.2026 | Pruning ekspertów MoE (routing mass) → specjalista tłumaczenia; 50–75% ekspertów usuwalne | MoE experts | Kontrole odwrócone | **Tak** | Nie | Częściowo | Bliskie projektowi UCT; specjalista językowy z MoE |
| 40 | EfficientXpert | arXiv 2025 | Pruning domenowy (health/legal) trudniejszy niż ogólny; ForeSight mask + LoRA | weights | Nie | **Tak** | Nie | Nie | Kolejny domain-expert baseline |
| 41 | You Had One Job (TAQ) | arXiv 11.2025 | Per-task alokacja bitów z hidden states; profile wrażliwości różnią się między zadaniami | layers | Częściowo | Tak | Nie | Częściowo | „Representation → task-specific compression” po stronie kwantyzacji |

---

# 4. Co jest już definitywnie zajęte

## A. „Znajdź domain neurons i zachowaj je”
Za blisko: Cus-Prun, task-aware pruning, SEAP, FineScope, Capability Localization.

## B. „Zrób z generalisty małego eksperta”
Za blisko: D-Pruner, Cus-Prun, TrimLLM, PPC-GPT, SubspacePath Pruner, EfficientXpert, Translation Specialists (MoE).

## C. „Ile parametrów potrzebuje zadanie?”
Za blisko: **LLM-Sieve**.

## D. „Shared core + domain module”
Za blisko: KOFF, częściowo task-aware pruning.

## E. „Capabilities/domains są modularne”
Za blisko: Capability Localization, Compressed Sensing for Capability Localization, Modular Cognitive Architecture, language-specific neuron literature.

## F. „Domeny są dekodowalne, ale nie przyczynowo wycinalne na poziomie neuronów”
Za blisko: **Decodable But Not Detachable** (3 rodziny, 8 domen). Obecna teza repo (`docs/paper_result_index.md`: *„Domain-selective activations can be highly decodable without being compact causal levers”*) jako samodzielny paper jest już nieaktualna.

## G. „Kompresja szkodzi kompetencjom nierówno”
Za blisko: The Cost of Down-Scaling (fact recall vs ICL), The Asymmetric Harms of LLM Compression (head vs tail knowledge). Opisowo zajęte; mechanistyczne wyjaśnienie wolne.

---

# 5. Gdzie literatura pozostawia napięcie

### Wynik 1 — modularność istnieje
- Language-specific neurons: języki mają wyraźne komponenty.
- ICML 2026 capability localization: math/code zależą od małych subsets **głów**.
- Modular Cognitive Architecture: szerokie cognitive domains mają causal modularity.

### Wynik 2 — modularność nie zawsze istnieje na poziomie neuronów
`Decodable But Not Detachable`:
- academic subject domain można świetnie dekodować z aktywacji,
- ale brak selektywnych neuron shells,
- language/modality są znacznie bardziej detachable,
- **badali tylko neurony FFN**.

### Wynik 3 — mimo tego domain experts da się budować
D-Pruner, Cus-Prun, TrimLLM, SubspacePath itd. potrafią kompresować model pod scenariusz/domenę.

### Wynik 4 — lokalizacja często nie przewiduje skutku interwencji
- Hase i in.: causal tracing nie mówi, gdzie edycja działa.
- Lee i in.: unlearning w „prawdziwym” regionie ≈ losowy region tej samej wielkości.

To jest **silny prior przeciw naiwnej wersji hipotezy**. Jednocześnie sprawia, że pytanie jest ciekawe niezależnie od odpowiedzi.

### Wynik 5 — wstępne wyniki repo (TinyLlama, QA v1)
Głowy wybrane po selektywności domenowej przyczynowo szkodzą domenie docelowej, a neurony MLP (wysoce dekodowalne) nie, nawet przy dopasowanym budżecie. Neurony wybrane na tokenach odpowiedzi (MMLU) jednak szkodzą. **To jest most między #26 (głowy) a #27 (neurony)**: pytanie o granularność dla domen wiedzy pozostaje otwarte.

Powstaje więc pytanie:

> **Dlaczego pewne kompetencje/domeny są łatwe do wydzielenia jako mały specjalista, mimo że niekoniecznie są lokalne na poziomie pojedynczych neuronów — i czy lokalizacja na jakiejkolwiek skali w ogóle to przewiduje?**

---

# 6. Najbardziej defensywna luka po audycie

## Robocze pytanie

### **Does Localization Inform Compression?**

albo:

### **At What Scale Is Knowledge Compressibly Modular?**

Dla domeny/capability \(d\) definiujemy **specialist compression frontier**:

\[
F_d(b)=
\max_{S:\,\mathrm{Cost}(S)\le b}
Q_d(M_S)
\]

przy dodatkowym minimalnym constraint na „core”:

\[
Q_{\mathrm{core}}(M_S)\ge \tau.
\]

Minimalny budżet:

\[
B_d(\epsilon,\tau)
=
\min_S \mathrm{Cost}(S)
\]

przy:

\[
Q_d(M_S)\ge Q_d(M)-\epsilon,
\qquad
Q_{\mathrm{core}}(M_S)\ge \tau.
\]

**Non-target domains nie są chronione.**

W praktyce \(F_d\) jest tylko **oszacowaniem z dołu** uzyskanym konkretnym prunerem. Dlatego wnioski muszą być odporne na wybór prunera (§7a).

---

# 7. Prawdziwy claim nie może brzmieć „minimalny expert”

LLM-Sieve już szuka minimalnej pojemności dla zadania.

Nasz claim musi dotyczyć **wyjaśnienia tej pojemności przez organizację modelu**.

Dla każdej capability/domeny mierzymy modularność na kilku skalach:

\[
\mathcal M_d^{\mathrm{neuron}},
\quad
\mathcal M_d^{\mathrm{MLPgroup}},
\quad
\mathcal M_d^{\mathrm{head}},
\quad
\mathcal M_d^{\mathrm{layer}},
\quad
\mathcal M_d^{\mathrm{subspace/path}}.
\]

I testujemy:

\[
\mathcal M_d^{(g)}
\longrightarrow
B_d(\epsilon,\tau).
\]

Główne pytania:
1. **Czy modularity rzeczywiście przewiduje specialist compressibility?**
2. **Na jakiej granularności \(g\) jest to prawdziwe?**
3. **Czy właściwa granularność zależy od typu kompetencji?**
4. **Czy decodability bez causal modularity przewiduje compressibility?**
5. **Czy zależność generalizuje między model families i na held-out targets?**
6. **Czy modularity przewiduje coś ponad trywialne predyktory?**

---

# 7a. Wymagania metodologiczne (nowe — to recenzenci sprawdzą najpierw)

### 1. Brak cyrkularności
Jeżeli \(\mathcal M_d^{(g)}\) liczymy z ablacji komponentów na skali \(g\), a frontier z usuwania tych samych komponentów według tych samych score'ów, korelacja jest częściowo zagwarantowana z konstrukcji.

Wymóg:
- modularność mierzona **tanio i lokalnie** na splicie *discovery* (pojedyncze ablacje / małe podzbiory / aktywacje),
- frontier liczony **niezależnym** prunerem (Wanda i SparseGPT z kalibracją na domenie docelowej; opcjonalnie structured), oceniany na splicie *test*,
- główny wynik musi przetrwać zmianę prunera.

### 2. Predyktory bazowe
Model regresji \(B_d \sim \mathcal M_d^{(g)} + \text{kontrole}\), gdzie kontrole to co najmniej:
- bazowa accuracy / loss modelu na \(d\),
- częstość wiedzy (head vs tail; proxy: popularność encji / częstość w korpusie),
- decodability domeny z aktywacji,
- wrażliwość na globalny pruning (Wanda bez kalibracji domenowej),
- rozmiar i trudność zbioru ewaluacyjnego.

Claim brzmi: **modularność dodaje moc predykcyjną ponad te kontrole**.

### 3. Liczba targetów
4 domeny × 3 modele = 12 punktów to za mało do predykcji. Cel: **25–40 targetów**:
- przedmioty MMLU / MMLU-Pro (pogrupowane w spójne klastry, z wystarczającą liczbą przykładów),
- języki z Belebele (m.in. afrikaans, isiZulu, isiXhosa — zweryfikować listę; wspólne z projektem UCT),
- kompetencje: math (GSM8K), code (HumanEval/MBPP), ewentualnie ICL / retrieval z kontekstu.

### 4. Targety rozpinające oś modularności
Literatura już przewiduje, gdzie są targety:
- języki → modularne na poziomie neuronów (#27, #29),
- math/code → modularne na poziomie głów (#26),
- przedmioty akademickie → niemodularne na poziomie neuronów (#27), na poziomie głów nieznane.

Hipoteza jest przez to ostra i falsyfikowalna.

### 5. Held-out evaluation
- Fit relacji na części targetów, predykcja na held-out targetach (lub held-out typie targetu).
- Jeśli są ≥2 rodziny: fit na jednej, predykcja na drugiej.

### 6. Statystyka
- Bootstrap po przykładach i po losowych kontrolach.
- Kilka seedów kalibracji.
- Raportować rozmiar efektu, nie tylko p-value.

---

# 8. Potencjalnie mocny wynik

Możliwy wynik pozytywny:

\[
\text{decodability}
\not\Rightarrow
\text{compressibility},
\qquad
\text{neuron selectivity}
\not\Rightarrow
\text{compressibility},
\]

ale:

\[
\boxed{
\text{causal modularity at an appropriate structural scale}
\Rightarrow
\text{specialist compressibility}
}
\]

Np.:
- języki: neuron-level modular → łatwy specialist,
- medicine: słaba neuron-level modularity, ale silniejsza struktura head/pathway → nadal dobrze compressible,
- math: inny bottleneck scale,
- agriculture: rozproszona wiedza → większy specialist budget.

To byłoby odkrycie o **organizacji wiedzy**, a nie nowy pruning algorithm.

Możliwy wynik negatywny, **też publikowalny, jeśli rygorystyczny**:

> lokalizacja na żadnej badanej skali nie przewiduje specialist compressibility ponad trywialne predyktory (analogicznie do #30, #31).

Najgorszy scenariusz to wynik niejednoznaczny na małej próbie. Przed nim chroni §7a.

---

# 9. Co odróżniałoby to od najbliższych prac

## vs. LLM-Sieve
LLM-Sieve: ile parametrów potrzebuje pojedyncze zadanie?
My: **co w strukturze modelu wyjaśnia różnice w minimalnej potrzebnej pojemności między zadaniami/domenami i na jakiej skali?**

## vs. SubspacePath Pruner
SubspacePath: representation subspace → pathway dla konkretnego scenariusza.
My: **systematyczna zależność modularity ↔ achievable specialist compression frontier across domains/models/granularities.**

## vs. Cus-Prun / D-Pruner / EfficientXpert
Oni: skonstruuj lepszego domain experta.
My: **wyjaśnij, kiedy i dlaczego domain expert może być mały.**

## vs. Decodable But Not Detachable
Oni: co jest neuron-level detachable i dlaczego zależy to od training granularity? (tylko neurony FFN, bez kompresji)
My: **czy detachability/modularity na różnych skalach (głowy, grupy MLP, warstwy) determinuje praktyczną compressibility specjalisty?** Ich wynik daje nam gotową oś targetów.

## vs. Capability Localization / Compressed Sensing
Oni: czy capability ma causal local components?
My: **czy struktura causal support mówi, jak mały może być działający specialist?**

## vs. Does Localization Inform Editing/Unlearning?
Oni: lokalizacja → sukces edycji/unlearningu (odpowiedź głównie: nie).
My: **lokalizacja → sukces kompresji specjalisty**, na wielu skalach i wielu typach kompetencji.

## vs. Cost of Down-Scaling / Asymmetric Harms
Oni: kompetencje kompresują się nierówno (opis).
My: **mechanistyczne wyjaśnienie tej nierówności** (lub dowód, że lokalizacja jej nie wyjaśnia).

## vs. TASP
Oni: czy sygnał spektralny przewiduje wrażliwość modułów na zadanie, jako bramka dla metody pruningu.
My: czy organizacja przewiduje **rozmiar specjalisty per target**, jako pytanie naukowe.

---

# 10. Co musi wyjść, żeby to miało poziom ICML

Nie wystarczy:
- kolejny pruning score,
- +1–2 pp,
- jeden TinyLlama,
- jedna domena,
- jedna granularność,
- korelacja na kilkunastu punktach bez kontroli.

### Wersja realna na ICML 2027 (≈15 tygodni)
- **Modele:** 2 rodziny 7–8B (np. Qwen2.5-7B, Llama-3.1-8B); trzecia (Gemma) tylko jeśli starczy czasu.
- **Targety:** 25–40 (§7a.3).
- **Granularności:** attention heads, grupy MLP, warstwy.
- **Prunery do frontierów:** Wanda + SparseGPT z kalibracją na targecie, bez recovery fine-tuningu w głównym eksperymencie; recovery na podzbiorze jako robustness check.
- **Budżety:** 100, 90, 75, 60, 50, 40%.
- **Analiza:** regresja z kontrolami + held-out targets + transfer między rodzinami.

Szacunek kosztu: ~30 targetów × 6 budżetów × 2 prunery × 2 modele ≈ 720 runów prune+eval. Wanda na 7B to minuty, więc to kilka dni na jednym A100/H100. Wąskim gardłem jest ewaluacja, nie pruning.

### Wersja pełna (NeurIPS 2027)
3 rodziny, dodatkowo subspace/pathway, recovery fine-tuning, więcej języków, analiza skali modelu.

---

# 11. Werdykt novelty po audycie

## NIE jest nowe
- domain-specific pruning,
- customized expert extraction,
- minimal task-specific model,
- capability localization,
- shared vs task-specific parameters,
- subspace/pathway pruning,
- „decodable but not detachable” na poziomie neuronów,
- opis, że kompresja degraduje kompetencje nierówno.

## Nadal wygląda na niezapełnioną lukę
> **Systematyczne, causalne badanie, czy wewnętrzna modularność na różnych poziomach strukturalnych przewiduje osiągalną wielkość/krzywą kompresji specjalisty — ponad trywialne predyktory i na held-out targetach.**

Nie znalazłem pracy, która równocześnie:
1. mierzy mechanistic/causal modularity,
2. robi to na wielu granularnościach,
3. porównuje różne rodzaje kompetencji (domeny, kompetencje, języki),
4. wyznacza specialist compression frontier niezależnym prunerem,
5. sprawdza, czy modularity przewiduje ten frontier między modelami/targetami.

---

# 12. Ryzyko

1. **Pole porusza się bardzo szybko.** #27 i #33 pojawiły się w sierpniu 2026. Szczególnie bliskie: SubspacePath Pruner, LLM-Sieve, Decodable But Not Detachable, Modular Cognitive Architecture, TASP. Autorzy #27 mogą naturalnie zrobić krok „detachable → compressible”.
2. **Obecna teza repo jest już opublikowana** (#27). Wyniki TinyLlama mają wartość tylko jako pilot i most do pytania o granularność.
3. **Hipoteza może się nie potwierdzić** (prior z #30, #31). Trzeba zaplanować paper tak, żeby wynik negatywny też był publikowalny.
4. **Recenzent: „frontier zależy od prunera, nie od domeny”.** Obrona: dwa niezależne prunery + robustness check z recovery.
5. **Recenzent: „to tylko baseline accuracy / częstość wiedzy”.** Obrona: §7a.2.
6. **Czas.** ~15 tygodni do projektowanego deadline'u, repo obecnie tylko na TinyLlamie.

Przed submission konieczny jest jeszcze jeden audit w grudniu/styczniu.

---

# 13. Pierwszy pilot — zanim budujemy paper

Nie pisać nowego prunera. Nie robić wielkiego refaktoru przed pilotem.

### Model
1 model 7–8B (np. Qwen2.5-7B lub Llama-3.1-8B). TinyLlama tylko do smoke testów.

### Targety (~15)
- 6–8 przedmiotów / klastrów MMLU (w tym medicine, astronomy, agriculture-like, law),
- 2–3 języki (Belebele),
- math, code.

### Krok 1 — organization (discovery split)
Mierzyć:
- activation decodability,
- selectivity,
- causal ablation (pojedyncze komponenty + top-k vs layer-matched random),
- overlap między targetami,

na:
- attention heads,
- grupach MLP.

### Krok 2 — specialist frontier (test split, niezależny pruner)
Wanda (i jeśli czas: SparseGPT) z kalibracją na targecie, budżety 90, 75, 60, 50%. Mierzyć target quality, non-target quality, core quality.

### Krok 3 — najważniejszy wykres
- Oś X: **dense-model causal modularity** (osobno per granularność).
- Oś Y: **minimal specialist budget / AUC frontieru**.
- Obok: ta sama zależność dla predyktorów bazowych (accuracy, decodability, globalna wrażliwość).

### Decyzja
- **GO:** modularity na którejś skali przewiduje frontier wyraźnie lepiej niż predyktory bazowe, a najlepsza skala różni się między typami targetów.
- **PIVOT (negatywny framing):** żadna skala nie przewiduje ponad predyktory bazowe, ale efekt jest wyraźnie zmierzony → „Localization does not inform compression”.
- **NO-GO na ICML:** wynik niejednoznaczny lub za szumny → pełna wersja na NeurIPS 2027.

---

# 14. Harmonogram

ICML 2027 (CFP jeszcze nieopublikowany, projekcja): abstrakt ~16.01.2027, paper ~22.01.2027 (AoE). Lokalizacja: Ameryka Południowa.

| Okres | Cel |
|---|---|
| do ~25.10 | Minimalny pipeline: target spec, component IDs, head/MLP-group masking dla modelu 7–8B, Wanda z kalibracją na targecie, ewaluacja per target |
| do ~20.11 | Pilot (§13) + **decyzja go/no-go** |
| grudzień | Skalowanie: 2. model, 25–40 targetów, 2. pruner, kontrole, held-out; **re-audit literatury** |
| 1.01–22.01 | Pisanie, figury, appendix, ostatni re-audit |
| fallback | NeurIPS 2027 (maj) z wersją pełną |

---

# 15. Roboczy tytuł po audycie

Najostrzejszy:

### **Does Localization Inform Compression? Functional Modularity and the Size of LLM Specialists**

Bezpieczny:

### **At What Scale Is Knowledge Compressibly Modular in Large Language Models?**

Alternatywy:

### **From Functional Modularity to Compact Specialists in Large Language Models**

### **What Makes Knowledge Extractable? Linking Functional Modularity to Specialist Compression**

Nie używać claimu „smallest viable specialist” jako głównej novelty — LLM-Sieve jest za blisko.

---

# 16. Najbliższe źródła

- SparseGPT — ICML 2023: https://proceedings.mlr.press/v202/frantar23a.html
- LLM-Pruner — NeurIPS 2023: https://arxiv.org/abs/2305.11627
- Wanda — ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/14c856c7a41297804de4c4890e846b25-Abstract-Conference.html
- Sheared LLaMA — ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/160adf2dc118a920e7858484b92a37d8-Abstract-Conference.html
- FLAP — AAAI 2024: https://ojs.aaai.org/index.php/AAAI/article/view/28960
- SliceGPT — ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/316648eb8b4ffb6010f531b07848c300-Abstract-Conference.html
- D-Pruner — Findings NAACL 2024: https://aclanthology.org/2024.findings-naacl.91/
- Language-Specific Neurons — ACL 2024: https://aclanthology.org/2024.acl-long.309/
- Capability Localization — ICLR 2025: https://arxiv.org/abs/2502.20992
- Cus-Prun — Findings ACL 2025: https://aclanthology.org/2025.findings-acl.1201/
- TrimLLM — ACL 2025: https://aclanthology.org/2025.acl-long.33/
- FineScope — arXiv 2025: https://arxiv.org/abs/2505.00624
- LLM-Sieve — arXiv 2025: https://arxiv.org/abs/2505.18350
- Cross-Lingual Generalization and Compression — ACL 2025: https://aclanthology.org/2025.acl-long.661/
- PPC-GPT — EMNLP 2025: https://aclanthology.org/2025.emnlp-main.747/
- Let LLM Tell What to Prune and How Much — ICML 2025: https://proceedings.mlr.press/v267/yang25p.html
- SEAP — AAAI 2026: https://ojs.aaai.org/index.php/AAAI/article/view/40463
- MixCal — EACL 2026: https://aclanthology.org/2026.eacl-long.347/
- NeuronMoE — EACL 2026: https://aclanthology.org/2026.eacl-long.117/
- Compressed Sensing for Capability Localization — ICML 2026: https://proceedings.mlr.press/v306/bair26a.html
- SubspacePath Pruner — ICML 2026: https://proceedings.mlr.press/v306/zhiren26a.html
- KOFF — arXiv 2026: https://arxiv.org/abs/2605.29075
- Capability-Guided Compression — arXiv 2026: https://arxiv.org/abs/2603.16440
- Decodable But Not Detachable — arXiv 2026: https://arxiv.org/abs/2608.10214
- Modular Cognitive Architecture Emerges — arXiv 2026: https://arxiv.org/abs/2608.13567
- CRANE — arXiv 2026: https://arxiv.org/abs/2601.04664

Nowe w wersji 2:
- Does Localization Inform Editing? (Hase i in.) — NeurIPS 2023: https://arxiv.org/abs/2301.04213
- Does Localization Inform Unlearning? (Lee i in.) — EMNLP 2025: https://arxiv.org/abs/2505.16252
- The Cost of Down-Scaling Language Models — arXiv 2023: https://arxiv.org/abs/2310.04680
- The Asymmetric Harms of LLM Compression — arXiv 2026: https://arxiv.org/abs/2608.19670
- Are Neural Nets Modular? (Csordás i in.) — ICLR 2021: https://people.idsia.ch/~csordas/are_neural_networks_modular.pdf
- Emergent Modularity in Pre-trained Transformers — Findings ACL 2023: https://arxiv.org/abs/2305.18390
- Gradient Routing — arXiv 2024: https://arxiv.org/abs/2410.04332
- TASP: Task-Aware Spectral Pruning — arXiv 2026: https://arxiv.org/abs/2609.29499
- Less is MoE (Fisher-MoE) — arXiv 2026: https://arxiv.org/abs/2606.05538
- Extracting Small Translation Specialists — arXiv 2026: https://arxiv.org/abs/2605.28042
- EfficientXpert — arXiv 2025: https://arxiv.org/abs/2511.19935
- You Had One Job (TAQ) — arXiv 2025: https://arxiv.org/abs/2511.06516
- Projekcja terminów ICML 2027: https://www.opencurious.com/ai-conference-deadlines/icml-2027
