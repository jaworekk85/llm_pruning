# ICML 2027 — Novelty Audit
## Knowledge organization → selective specialization/compression

**Aktualizacja: 13 października 2026 — audyt kompozycji i kompresji (§20)**
(wersja 2 z 9.10 zachowana jako baza; aktualne korekty i werdykt: §18 oraz
`docs/literature_novelty_review_2026-10-11.md`)

**Werdykt:** ogólne „lokalizacja → kompresja” jest zajęte. Niezależna predykcja
specialist frontierów na nowych targetach pozostaje kandydatem na wkład, z silnymi
precedensami w TASP, CGC, TaCQ i DiscoGP. Nie deklarować pierwszeństwa bez zastrzeżeń.
§17 dokumentuje przerwany pierwszy etap; §18 i powiązany raport go aktualizują.

**Priorytety z 12.10:** pierwszy kandydat do sprawdzenia to utrata kompozycji
umiejętności podczas pruningu. Poniższy audyt predykcji modularity→compressibility
pozostaje przydatny, ale nie jest zamrożonym głównym protokołem. Aktualne decyzje:
[research_strategy.md](docs/research_strategy.md); nowe precedensy i granice: §20
i [audyt kompozycji z 13.10](docs/composition_novelty_review_2026-10-13.md).

**Zakres:** ten dokument dotyczy wyłącznie paperu ICML 2027. Projekt UCT (języki) może korzystać z tych samych narzędzi i częściowo z tych samych celów (języki jako targety). Projekt Waikato (streaming) jest osobnym projektem i nie jest częścią tego paperu.

---

## 0. Co się zmieniło w wersji 2

1. **Weryfikacja źródeł w wersji 2.** Wcześniejszy audyt deklarował sprawdzenie pozycji #16, #19, #20, #21, #26, #27, #28, #29. Nie oznacza to, że każdy opis metod i wniosek o nowości został ponownie potwierdzony; ograniczony re-audyt opisano w §17.
2. **Dodano 12 brakujących prac** (#30–#41). Najważniejsza luka w wersji 1: brak linii *„Does localization inform …?”*, która jest jednocześnie głównym ryzykiem i najlepszym framingiem.
3. **Dawny framing repo ma bezpośredni precedens.** *Decodable But Not Detachable* (08.2026) bada rozdział dekodowalności i selektywnej konieczności. To ogranicza nowość dawnej tezy; nasze małe próby/fixtures nie ustanawiają równoważnego wyniku naukowego. Patrz §4F i §12.
4. **Dodano wymagania metodologiczne** (§7a): cyrkularność, predyktory bazowe, liczba targetów, held-out targets, niezależne prunery.
5. **Przeprojektowano pilot** (§13) i dodano harmonogram z decyzją go/no-go (§14).

---

## 1. Roboczy problem badawczy

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
4. **„Domeny są dekodowalne, ale nie dają się wyciąć na poziomie neuronów”** — *Decodable But Not Detachable* (08.2026). Bezpośredni precedens dla dawnego framingu repo; wynik jest zależny od ustawienia i interwencji.

Znane jest też, że **kompresja degraduje kompetencje nierówno** (fact recall vs ICL; wiedza head vs tail). Istnieją już propozycje wyjaśnień przez redundancję, strukturę cech i kompensację; nie twierdzić, że cała luka mechanistyczna pozostaje wolna.

Roboczy kierunek, wymagający odróżnienia od precedensów w §18:

> **Czy miary organizacji funkcjonalnej w gęstym modelu dodają wartość predykcyjną dla jakości niezależnie przyciętego specjalisty na nowych targetach — i czy zależy to od granularności?**

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
- **✓**: historyczne oznaczenie potwierdzenia źródła w wersji 2 lub aktualizacja 11.10; nie oznacza pełnej recenzji pracy. Faktyczny poziom lektury bieżącego audytu zapisano w dzienniku §18.

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
| 21 ✓ | Capability-Guided Compression | arXiv 03.2026 | SAE density → budżety i teoria progów; GPT-2 Medium, negatywne wyniki | heads/components | **Tak: singleton ablations** | Nie | Nie | Test predykcyjny, wynik negatywny | **Już bada descriptor → ablation impact; wysoki priorytet** |
| 22 | NeuronMoE | EACL 2026 | Language-specific neurons kierują allocation experts | neurons/layers | Nie | Modular experts | Nie | **Tak dla języków** | Ważne dla projektu UCT |
| 23 | Language-Specific Neurons | ACL 2024 | Mała populacja neuronów specyficznych językowo | neurons | Steering/masking | Nie | Nie | Częściowo | Silna modularność językowa |
| 24 | Capability Localization | ICLR 2025 | Capabilities, nie pojedyncze fakty, mogą być lokalizowane | neurons | Tak/decoupling | Nie | Nie | Częściowo | Localization sama w sobie zajęta |
| 25 | Cross-Lingual Generalization and Compression | ACL 2025 | Language-specific → shared semantic neurons | neurons | Steering | Nie | Nie | **Tak** | Shared/specific representations |
| 26 ✓ | Compressed Sensing for Capability Localization | ICML 2026 | 5 głów może obniżyć math/code nawet o 60% przy małym wpływie na inne zadania; Llama/Qwen 1–14B | attention heads | **Tak** | Nie | Nie | Częściowo | Bardzo mocne capability localization **na poziomie głów** |
| 27 ✓ | Decodable But Not Detachable | arXiv 08.2026 | Dekodowalność FFN, damage matrices i INT4 down-projection w grupach po 128 kanałów (§4.6); przeplatanie neuronów utrudnia wybór core-only grup | FFN neurons + quantization groups | **Tak** | Nie | Test kwantyzacji | Training granularity: interpretacja autorów | **Kluczowy precedens; bada faktycznie kompresję grupową. Nie przypisywać mu uniwersalnego prawa o pretrainingu** |
| 28 ✓ | Modular Cognitive Architecture Emerges | arXiv 2026 (Han, Andreas, Fedorenko, de Varda) | 46 zadań, 4 cognitive domains; nakładanie się neuronów odzwierciedla sieci w mózgu | MLP neurons/circuits | **Tak** | Nie | Nie | Częściowo | Modularność na szerszych cognitive domains |
| 29 ✓ | CRANE | arXiv 01.2026 | Causal language-specific neurons; „selective but non-exclusive” | neurons | **Tak** | Nie | Nie | Częściowo | Potwierdza causal language modularity |
| 30 | **Does Localization Inform Editing?** (Hase i in.) | NeurIPS 2023 | Causal tracing wnosi mało ponad wybór warstwy do predykcji sukcesu edycji w badanych ustawieniach | layers | **Tak** | Nie | Nie | — | **Bezpośredni wzorzec testu dodatkowej wartości predykcyjnej** |
| 31 | **Does Localization Inform Unlearning?** (Lee i in.) | EMNLP 2025 | Unlearning w prawdziwym regionie daje tylko marginalny zysk vs losowy region tej samej wielkości; wiele alternatywnych regionów | value vectors | **Tak** | Nie | Nie | — | **Główne ryzyko dla hipotezy i najlepszy framing paperu** |
| 32 | The Cost of Down-Scaling LMs (Jin i in.) | arXiv 2023 | >30% redukcji szkodzi fact recall; 60–70% redukcji oszczędza ICL | model/weights | Nie | Nie | Nie | **Tak, empirycznie** | Różna kompresowalność kompetencji jest znana; także wyjaśnienia mają precedensy |
| 33 | The Asymmetric Harms of LLM Compression | arXiv 08.2026 | 3 LLM × 11 metod; względna utrata wiedzy head/tail zależy od miary i datasetu | — | Nie | Nie | Nie | Opisowo | Popularność jako proxy; nie pomiar rzeczywistej częstości w nieznanym pretrainingu |
| 34 | Are Neural Nets Modular? (Csordás i in.) | ICLR 2021 | Funkcjonalna modularność przez differentiable weight masks | weights | **Tak** | Nie | Nie | Częściowo | Obowiązkowy precedens metodologiczny dla \(\mathcal M_d\) |
| 35 ✓ | Emergent Modularity in Pre-trained Transformers | Findings ACL 2023 | Funkcjonalni eksperci, perturbacje i restricted routing + fine-tuning GLUE (§5) | neurons/experts | **Tak: perturbacje** | Restricted experts + recovery | Nie | Częściowo | Silniejszy precedens niż same skupiska; nie mylić z retencją bez treningu |
| 36 | Gradient Routing | arXiv 2024 | Maskowanie gradientów lokalizuje zdolności w wybranych regionach już w treningu | regions | **Tak** | Nie | Nie | Częściowo | Modularność jako własność treningu; wspiera wniosek #27 |
| 37 | TASP (Task-Aware Spectral Pruning) | arXiv 2026 | Prediction ablation damage; **Appendix R: diagnostyka → retencja na nowych checkpointach** | heads + SwiGLU groups | Kalibracja ablacjami | **Tak/routed** | Nie | **Tak: test predykcyjny** | Najbliższy precedens; odróżnić niezależnych prunerów i nowe targety |
| 38 | Less is MoE (Fisher-MoE) | arXiv 06.2026 | Usunięcie 12 z ~1,35 mln wymiarów FFN niszczy GSM8K, a wiedza faktograficzna zostaje | FFN intermediate dims (MoE) | **Tak** | Tak | Nie | Częściowo | Wspiera tezę „granularność zależy od kompetencji”; konkurencja w MoE |
| 39 | Extracting Small Translation Specialists | arXiv 05.2026 | Pruning ekspertów MoE (routing mass) → specjalista tłumaczenia; 50–75% ekspertów usuwalne | MoE experts | Kontrole odwrócone | **Tak** | Nie | Częściowo | Bliskie projektowi UCT; specjalista językowy z MoE |
| 40 | EfficientXpert | arXiv 2025 | Pruning domenowy (health/legal) trudniejszy niż ogólny; ForeSight mask + LoRA | weights | Nie | **Tak** | Nie | Nie | Kolejny domain-expert baseline |
| 41 ✓ | You Had One Job (TAQ) | arXiv 2025; v4 06.2026 | Representation-guided mixed precision + singleton quantization oracle; 87.5% przydziałów wspólnych między zadaniami w raporcie autorów | layers | **Tak: lokalna perturbacja** | Tak | Nie | Częściowo | Representation → compression zajęte; wspólna wrażliwość warstw jako kontrola |
| 42 | TaCQ | COLM 2025 camera-ready wg arXiv | Gradient + symulowana kwantyzacja → chronione wagi | weights | Saliency, nie pełna mapa konieczności | Tak | Budżet bitów | Transfer kalibracji | Localization-guided compression już istnieje |
| 43 | DiscoGP | EMNLP 2025 | Joint weight/edge pruning; samodzielna fidelity | weights + graph edges | **Tak: zero ablation** | Tak | Sparse sheaf | Częściowo | Małe funkcjonalnie wystarczające moduły są zajęte |
| 44 | AlphaPruning | arXiv 2024, tekst sprawdzony | Spektralna alokacja sparsity dla Magnitude/Wanda/SparseGPT | layers/blocks | Nie | Nie | Nie | Alokacja | Struktura → budżety wielu prunerów jest zajęta |
| 45 | ShadowLLM | EMNLP 2024 | Contextual importance predictor | heads/neurons | Gradient proxies | Dynamic sparsity | Nie | Per input | Predykcja ważności to istniejąca linia |
| 46 | Perturbation–Absorption | arXiv 06.2026 | Interwencje i kompensacja błędu korygują alokację | layers | **Tak: perturbation injection** | Nie | Nie | Mechanizm propagacji | Konkurencyjne wyjaśnienie ponad lokalną ważność |
| 47 | Configurable Foundation Models | arXiv 2024 | Framework „bricks”, functionality scores i neuron masking | neurons/modules | Masking experiments | Framework | Nie | Częściowo | Ogólny framework modularności nie jest nowym wkładem |
| 48 | Impact of Calibration Data | ACL 2024 | Powtarzane pule kalibracyjne w 1800 modelach; downstream variability | calibration | Nie | Nie | Nie | Confounder | Kontrola kalibracji jest konieczna |
| 49 | Beware of Calibration Data | ICLR 2025 | Źródła danych, ilość i zgodność z pretrainingiem; synthesis | calibration | Nie | Nie | Nie | Confounder | Więcej przykładów nie usuwa różnic między źródłami w badanym ustawieniu |
| 50 | UniComp | arXiv v6 08.2026; tylko abstrakt | Unified pruning/quantization/distillation i capability benchmark | model | Niezweryfikowane | Niezweryfikowane | Niezweryfikowane | Task calibration | Benchmark wielu zdolności już istnieje; nie oceniono pełnej metodologii |
| 51 | OCP | ACL 2026; tylko abstrakt | Outlier-centric input probes i online structured pruning | channels/layers | Niezweryfikowane | Dynamic | Niezweryfikowane | Input adaptive | Odróżnić statycznego specjalistę od dynamicznego wykonania |

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
Za blisko: **Decodable But Not Detachable**. Dawny framing repo (`docs/paper_result_index.md`: *„Domain-selective activations can be highly decodable without being compact causal levers”*) nie wystarcza do nowości samodzielnego paperu; nasze próby nie są reprodukcją wyników tej pracy.

## G. „Kompresja szkodzi kompetencjom nierówno”
Za blisko: The Cost of Down-Scaling (fact recall vs ICL), The Asymmetric Harms of LLM Compression (head vs tail knowledge). Istnieją także wyjaśnienia redundancy/compensation (CGC, Perturbation–Absorption); szerokiego „dlaczego” nie traktować jako wolnego.

## H. „Interpretowalny mały obwód jest dobrym specjalistą”
Za blisko: TaCQ, DiscoGP, SubspacePath. Konieczność, wystarczalność i fizyczna kompresja wymagają odrębnych pomiarów.

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
Oni: FFN detachability oraz ograniczenie selektywnej kwantyzacji grupowej (§4.6).
My: **czy miary organizacji przewidują wyniki niezależnego przycinania?** Grupy i sama kompresja nie stanowią odróżnienia; raport w §18 precyzuje węższy test.

## vs. Capability Localization / Compressed Sensing
Oni: czy capability ma causal local components?
My: **czy struktura causal support mówi, jak mały może być działający specialist?**

## vs. Does Localization Inform Editing/Unlearning?
Oni: lokalizacja → sukces edycji/unlearningu (odpowiedź głównie: nie).
My: **lokalizacja → sukces kompresji specjalisty**, na wielu skalach i wielu typach kompetencji.

## vs. Cost of Down-Scaling / Asymmetric Harms
Oni: kompetencje kompresują się nierówno (opis).
My: **wartość predykcyjna miar funkcjonalnych ponad prostsze wyjaśnienia**; korelacja sama nie ustanawia mechanizmu ani dowodu jego nieistnienia.

## vs. TASP
Oni: także predykcja retencji między nowymi checkpointami (Appendix R).
My: **przenoszenie predykcji na nowe targety i niezależne prunery**, z porównaniem baseline-only versus baseline+organization. Samo „pytanie naukowe zamiast metody” nie wystarczy.

---

# 10. Co musi wyjść, żeby to miało poziom ICML

Ta sekcja opisuje wcześniejszego kandydata predykcyjnego; nie narzuca liczby modeli,
targetów ani granularności aktualnemu screenowi kompozycji. Plan wykonawczy: §13.

Nie wystarczy:
- kolejny pruning score,
- +1–2 pp,
- jeden TinyLlama,
- jedna domena,
- jedna granularność,
- korelacja na kilkunastu punktach bez kontroli.

### Wcześniejsza propozycja zakresu kandydata predykcyjnego
- **Modele:** 2 rodziny 7–8B (np. Qwen2.5-7B, Llama-3.1-8B); trzecia (Gemma) tylko jeśli starczy czasu.
- **Targety:** 25–40 (§7a.3).
- **Granularności:** attention heads, grupy MLP, warstwy.
- **Prunery do frontierów:** Wanda + SparseGPT z kalibracją na targecie, bez recovery fine-tuningu w głównym eksperymencie; recovery na podzbiorze jako robustness check.
- **Budżety:** 100, 90, 75, 60, 50, 40%.
- **Analiza:** regresja z kontrolami + held-out targets + transfer między rodzinami.

Przykładowa liczba konfiguracji: 30 targetów × 6 budżetów × 2 prunery × 2 modele = 720 prune+eval, przed powtórzeniami seedów. Czas i pamięć zmierzyć na serwerze; ta liczba nie uzasadnia prognozy „kilka dni”.

### Opcjonalne rozszerzenie po walidacji wkładu
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

## Kandydat na wkład — warunkowy werdykt po re-audycie
> **Czy discovery-only miary organizacji funkcjonalnej dodają wartość predykcyjną dla krzywych niezależnych prunerów na held-out targetach, ponad jakość bazową i wrażliwość na pruning?**

W odczytanych fragmentach nie znaleziono pełnego odpowiednika testu łączącego:
1. mierzy mechanistic/causal modularity,
2. robi to na wielu granularnościach,
3. porównuje różne rodzaje kompetencji (domeny, kompetencje, języki),
4. wyznacza specialist compression frontier niezależnym prunerem,
5. sprawdza, czy modularity przewiduje ten frontier między modelami/targetami.

TASP, CGC, TaCQ, DiscoGP i AlphaPruning zajmują znaczną część tego framingu.
Lista pięciu cech sama nie dowodzi nowości przez ich kombinację. Konkretny wkład
musi wynikać z nowego testu i użytecznych wyników; szczegóły źródeł i luk: §18.

---

# 12. Ryzyko

1. **Pole porusza się bardzo szybko.** Najbliższe: TASP (już predykcja między modelami), CGC, TaCQ, DiscoGP, SubspacePath i LLM-Sieve. #27 już bada ograniczenie kompresji grupowej; wcześniejszy audyt zaniżał overlap.
2. **Dawny framing repo ma bezpośredni precedens** (#27). TinyLlama/random fixtures służą przygotowaniu pilota; nie uzasadniają nowego claimu naukowego.
3. **Hipoteza może się nie potwierdzić** (prior z #30, #31). Trzeba zaplanować paper tak, żeby wynik negatywny też był publikowalny.
4. **Recenzent: „frontier zależy od prunera, nie od domeny”.** Obrona: dwa niezależne prunery + robustness check z recovery.
5. **Recenzent: „to tylko baseline accuracy / częstość wiedzy”.** Obrona: §7a.2.
6. **Czas/zasoby.** Brak pretrained walidacji nowych pipeline'ów; koszt screenów
   trzeba zmierzyć na serwerze. Oficjalne terminy ICML 2027 pozostają niezweryfikowane.

Przed submission konieczny jest jeszcze jeden audit w grudniu/styczniu.

---

# 13. Pierwszy pilot — zanim budujemy paper

Aktualizacja 12.10: plan operacyjny ma jedno miejsce —
[docs/projects/icml2027.md](docs/projects/icml2027.md). Nie budować nowego prunera
ani pełnego frameworku przed screenem.

1. Pierwszy kandydat: A, B i A→B na powiązanych przypadkach, ze staged/oracle
   controls; jeden pretrained model 7–8B i mała siatka Wandy.
2. Oddzielić utratę umiejętności składowych i propagację błędów od dodatkowej
   utraty kompozycji. Splitować grupy przypadków/źródeł, nie ich warianty osobno.
3. Po sygnale: causal support i ochrona/przywracanie przy równym faktycznym koszcie,
   test na nowych kombinacjach i innym modelu/prunerze.
4. Dawna predykcja organization→frontier pozostaje kandydatem dodatkowym; wymaga
   własnej liczby targetów i testu held-out. Kilka task bundles jej nie potwierdzi.

**GO do skalowania:** precyzyjny sygnał ponad prostsze kontrole, wykonalny koszt
i konkretne odróżnienie od najbliższych prac. To nie jest jeszcze GO do akceptacji ICML.
Przy wyniku słabym/niejednoznacznym modyfikujemy pytanie; nie ogłaszamy automatycznie
paperu negatywnego ani zmiany konferencji. Głównym celem pozostaje ICML 2027.

---

# 14. Harmonogram

Terminy submission ICML 2027 nie zostały tu zweryfikowane; 12.10 narzędzie nie
odczytało oficjalnej strony 2027. Nie traktować dawnych projekcji jako deadline'ów.
Poniżej cele wewnętrzne, zależne od dostępu do serwera; aktualizować po pomiarze kosztu.

| Okres | Cel |
|---|---|
| do ~25.10 | Domknąć kontrakt linked tasks i minimum danych/evaluatora; nie odbudowywać dostępnych masek/Wandy |
| do ~20.11 | Pretrained baseline + screen, prostsze kontrole i decyzja o kierunku (§13), jeśli GPU dostępne |
| po GO | Lokalizacja/mitigacja, 2. model/pruner, nowe kombinacje, re-audit dokładnego claimu |
| przed potwierdzonym submission | Pisanie/figury/appendix, test zamrożonego protokołu i odświeżenie literatury |
| UCT/Waikato | Przygotowanie zakresu z partnerami może działać równolegle; osobne pytania i harmonogramy |

---

# 15. Roboczy tytuł po audycie

Poniższe tytuły dotyczą wcześniejszego kandydata predykcyjnego. Tytuł głównego
paperu ustalić po wyborze wkładu i wynikach; nie zamrażać go na podstawie niszy.

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

# 17. Ograniczony re-audyt — 11.10.2026, etap 1

**Archiwalny checkpoint przerwanego etapu. Został uzupełniony przez §18 i powiązany raport; poniższe „jeszcze nie zweryfikowane” odnosi się do chwili przerwania.**

**Stan: częściowy, zakończony na limicie; nie pełny audyt nowości.**
Budżet wynosił 10 000 tokenów; automatyczny licznik zgłosił 17 750 przed
zapisaniem podsumowania, ponieważ odpowiedzi narzędzi przekroczyły oczekiwany rozmiar.
Nie wykonano systematycznego przeglądu cytowań ani pełnego porównania metod/wyników.

Pytanie: czy niezależne pomiary organizacji funkcjonalnej przewidują jakość lub
minimalny obserwowany budżet specjalisty uzyskanego innym prunerem, ponad baseline
quality/sensitivity, na nowych targetach? Brak znalezionej pracy nie dowodzi nowości.

## Potwierdzone źródła i rzeczywisty poziom lektury

| Praca | Co sprawdzono | Znaczenie dla hipotezy |
|---|---|---|
| [LLM-Sieve, v2](https://arxiv.org/html/2505.18350v2) | Metadane, abstrakt oraz dostępny fragment wstępu i related work; pełnego eksperymentalnego porównania nie wykonano | Autorzy łączą task compression z nierówną organizacją wiedzy i bottleneckami. Ogólny claim „organizacja → kompresja” jest zajęty; oddzielić niezależną predykcję od diagnozy otrzymanej podczas optymalizacji. |
| [SubspacePath Pruner](https://proceedings.mlr.press/v306/zhiren26a.html) | Oficjalna strona ICML 2026 i abstrakt; PDF nie został odczytany | Reprezentacyjne dopasowanie domen służy budowaniu budżetowanych ścieżek głów. Bardzo bliski precedens metody; nie ustalono, czy zawiera porównywalny test predykcyjny. |
| [Capability-Guided Compression](https://arxiv.org/abs/2603.16440) | Metadane i abstrakt; HTML niedostępny | Autor deklaruje teorię związku capability density z redundancją i progami kompresji. To bezpośrednie ryzyko dla szerokiego claimu predykcji, mimo negatywnego wyniku PPL na GPT-2 Medium. Założenia/dowody wymagają lektury. |
| [Decodable But Not Detachable](https://arxiv.org/abs/2608.10214) | Metadane i abstrakt; HTML zlokalizowany, szczegółowe metody nieweryfikowane | Dekodowalność nie zapewnia selektywnej konieczności; abstrakt wiąże przestrzenne przeplatanie neuronów z ograniczeniem selektywnej kwantyzacji grupowej. Nie traktować samego wyboru grup jako wystarczającej nowości. |
| [TASP](https://arxiv.org/html/2609.29499v1) | Tożsamość i dostępność pełnego HTML; bez szczegółowej lektury | Pozostaje pilnym kandydatem do porównania task-aware pruning, masek i budżetów. |
| [D-Pruner](https://aclanthology.org/2024.findings-naacl.91/) i [Cus-Prun](https://aclanthology.org/2025.findings-acl.1201/) | Potwierdzenie tytułów na oficjalnych stronach ACL | Potwierdzono tożsamość precedensów; szczegółów tabeli §3 nie re-audytowano. |

## Dodatkowe kandydaty pominięte w wersji 2 — jeszcze nie zweryfikowane

- [Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression](https://arxiv.org/abs/2504.07389): wynik wyszukania w arXiv; sam tytuł wskazuje bezpośredni precedens. Pierwszy priorytet: metody, przyczynowość obwodów, predykcja i transfer między zadaniami.
- [The Connectome of a Large Language Model](https://openreview.net/pdf/9a49e30eea369ea5330cef29c63a69773fa3409a.pdf): znaleziony PDF OpenReview; fragment wyszukiwarki mówi o związku modularności z compressibility. **Potencjalnie najgroźniejszy kontrprzykład**, ale nie odczytano kontekstu: może chodzić o kompresję grafu cech, a nie wag modelu. Najpierw rozstrzygnąć ten punkt.
- [Sapling: Successive Adaptation and Compression with Layer Dropping for LLMs](https://openreview.net/pdf?id=vzvCaYFTLq): fragment pierwotnego PDF opisuje domain specialization oparte na obserwowanej lokalizacji wiedzy. Potwierdzić aktualną wersję/status publikacji i eksperymenty; nazwa pochodzi z wersji under review.
- [Modular Representation Compression](https://arxiv.org/abs/2604.18146): kandydat z wyszukiwarki dotyczący kompresji reprezentacji dla rekomendacji. Odróżnić koszt embeddingów od wielkości specjalistycznego LLM; niższy priorytet.

## Werdykt roboczy i następny etap

Szerokiej tezy „lokalizacja/modularność pomaga kompresji” nie wolno przedstawiać
jako nowej. Węższa predykcja frontierów niezależnego prunera na held-out targetach
pozostaje **kandydatem na wkład**, ale etap 1 nie potwierdza wolnej luki.
Następnie: (1) odczytać TCQ i znaczenie compressibility w Connectome, (2) porównać
metody/wyniki CGC, SubspacePath, LLM-Sieve i TASP, (3) sprawdzić ich cytowania.
W tabeli porównawczej oznaczać osobno: pomiar przyczynowy, niezależność prunera,
kontrole baseline, różne granularności i predykcję na niewidzianych targetach.
Nie zamrażać claimu ICML przed wykonaniem tych kroków; uniwersalne narzędzia repo
pozostają przydatne niezależnie od rozstrzygnięcia nowości.

# 18. Re-audyt metod i wyników — 11.10.2026

Raport: [literature_novelty_review_2026-10-11.md](docs/literature_novelty_review_2026-10-11.md).
Dziennik źródeł/zapytań: [literature_audit_20261011_sources.json](docs/literature_audit_20261011_sources.json).

Rozstrzygnięcia:
- TaCQ i DiscoGP to istotne, wcześniej pominięte precedensy localization-guided compression/sufficiency.
- TASP zawiera predykcję między checkpointami; CGC bada density–ablation i raportuje wynik negatywny. Nie sprowadzać ich do samego „framingu”.
- Connectome: w odczytanym fragmencie ratio dotyczy grafu cech, nie wag LLM. Pełny PDF pozostaje niedostępny.
- SubspacePath: przejrzano także README autorów; pełne wyniki paperu wymagają dostępu do PDF.
- AlphaPruning/ShadowLLM/Perturbation–Absorption poszerzają wymagane punkty odniesienia.
- TAQ v4 i prace kalibracyjne wymagają kontroli wspólnej wrażliwości warstw oraz źródła/seedów kalibracji. Starsze Emergent Modularity już zawiera restricted routing z fine-tuningiem.

**Decyzja:** warto wykonać mały falsyfikowalny pilot węższej predykcji na nowych
targetach/prunerach. Nie deklarować „pierwszego związku modularności z kompresją”.
Zakres nie obejmuje ponownej pełnej weryfikacji wszystkich 41 starszych pozycji,
reprodukcji ani wyczerpującego forward citation search. Przed submission odświeżyć
cytowania i zamknąć wskazane braki dostępu. Nowe źródła pierwotne są linkowane w raporcie.
Aktualny dziennik: 37 źródeł o różnych poziomach lektury i 40 zapytań; wykonano też
ukierunkowane szukanie powiązanych prac, bez twierdzenia o pełnym pokryciu cytowań.

# 19. Kandydat composition-under-pruning — kontrola precedensów 12.10.2026

Historyczny przegląd wstępny; rozszerzony audyt i aktualny werdykt: §20 poniżej.

To ukierunkowane sprawdzenie alternatyw, nie pełny audyt nowości nowej hipotezy.
- [Circuit Compositions](https://aclanthology.org/2025.acl-long.727/): sprawdzono
  oficjalny abstrakt i wyszukiwanie w PDF; samo łączenie obwodów nie jest nowe.
- [CoAx v3](https://arxiv.org/html/2607.01940v3): abstrakt i fragment §5/related work
  opisują conditional support, backup completion oraz transfer między modelami.
  Samo odkrycie ważności po ablacji nie odróżni naszego paperu.
- [ProxySPEX](https://arxiv.org/html/2505.17495): fragment eksperymentu Figure 10
  już porównuje interaction-aware pruning głów Llama-3.1-8B z marginalnym baseline'em.
- [TAGQuant](https://aclanthology.org/2026.eacl-industry.18/): oficjalny abstrakt;
  przestawianie kanałów dla grupowej kwantyzacji już istnieje.

Możliwy wkład musi wynikać z ustalenia przyczyny utraty kompozycji podczas kompresji
oraz użytecznej konsekwencji przy kontrolowanym koszcie. Nie ustalono, że taki efekt
występuje ani że dokładny test nie ma precedensu. Pełny audyt wybranego claimu jest
warunkiem skalowania. Współprace UCT/Waikato nie służą jako dodatkowy argument nowości.

# 20. Audyt kompozycji i kompresji — 13.10.2026

Wykonano 18 zapytań i przegląd 16 źródeł pierwotnych: 12 z lekturą istotnych
metod/wyników lub ograniczeń, jedno ze sprawdzeniem metod i trzy na poziomie
abstraktu. Raport oraz jawne braki pokrycia:
[composition_novelty_review_2026-10-13.md](docs/composition_novelty_review_2026-10-13.md),
[dziennik źródeł](docs/composition_audit_20261013_sources.json).
To audyt wybranego claimu; nie dowód braku precedensów ani reprodukcja.

**Korekta wstępnego obrazu:** prace o kwantyzacji już analizują wieloetapowe
rozumowanie i lokalizację jego uszkodzeń, a badania rzadkich obwodów zestawiają
zadania atomowe i złożone. Najbliższe nowe pozycje:
[PRISM](https://link.springer.com/article/10.1007/s10791-026-10580-5),
[Through a Compressed Lens v3](https://arxiv.org/html/2505.13963v3),
[Circuit Claims](https://arxiv.org/html/2607.18921) i
[CircuitLoRA §6](https://arxiv.org/html/2502.11812v2).
Sama utrata kompozycji, dodatkowe „mostki”, suma obwodów albo zmiana kwantyzacji
na Wandę nie stanowią jeszcze mocnego argumentu nowości.

**Decyzja:** zachować mały pilot. Kandydatem jest przyczynowo uzasadniona ochrona
kompozycji w niezależnie przyciętym modelu, z korzyścią na niewidzianych kombinacjach
przy tej samej zachowanej pojemności i kontrolowanym koszcie selekcji. Porównać z
sumą wsparcia A/B, metodami singleton/activation oraz conditional/interaction-aware.
Efekt i przewaga nadal nie są potwierdzone; nie obiecywać przyjęcia na ICML 2027.

Lokalny przegląd danych usunął skróty pozycji/nazwy i stałej różnicy dystraktora
w `count_affine_v2`. [Plan kontroli i precyzji](docs/composition_validation_plan.md)
oddziela 32-przypadkowy ekran wykonalności od potwierdzenia, określa kontrasty
sparowane, kontrolę wielokrotnych porównań i scenariusze liczebności. Analiza
potwierdzająca i dodatkowe kontrole promptów muszą być gotowe przed testem.

# 21. Przeprojektowanie benchmarku — 13.10.2026

Zestawy station-count v1/v2 pozostają ćwiczeniami inżynieryjnymi. Nowy kandydat
pomiarowy rozdziela lookup→lookup i lookup→dodawanie, zapewnia tę samą treść
źródłową dla A/B/zadania złożonego oraz powiązane kontrole zmian faktu istotnego,
dystraktora, kolejności i nazw. Wynik główny liczy tylko przypadki referencyjne;
kontrole nie powiększają liczby niezależnych przykładów. Szczegóły i kryteria
akceptacji: [composition_benchmark_design.md](docs/composition_benchmark_design.md).

Sprawdzono dodatkowo metodologię
[MuSiQue §3–4](https://aclanthology.org/2022.tacl-1.31/)
i fragmenty wprowadzenia/dyskusji skrótów
[SOCRATES](https://aclanthology.org/2025.findings-acl.205/).
To uzasadnienie kontroli i kandydaci na późniejszy transfer/fakty zamknięte,
nie import danych ani twierdzenie o nowości naszego benchmarku.
Zachowanie pretrained modelu i interwencje neuronowe pozostają niezweryfikowane.
