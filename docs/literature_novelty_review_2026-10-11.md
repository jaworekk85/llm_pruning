# Lokalizacja a kompresja — przegląd najbliższych metod, 11.10.2026

**Uwaga z 12.10:** raport ocenia kandydata predykcyjnego, który pozostaje analizą
dodatkową. Aktualny pierwszy screen dotyczy kompozycji pod pruningiem; priorytety:
[research_strategy.md](research_strategy.md), precedensy nowego kandydata: audyt §19.
Nie traktować poniższej decyzji jako aktualnego zamrożonego protokołu ICML.

## Werdykt

**Szeroki pomysł nie jest nowy. Wąski test predykcyjny pozostaje kandydatem na wkład.**
W odczytanych źródłach nie znaleziono pełnego odpowiednika następującego testu:
miary organizacji funkcjonalnej z discovery przewidują krzywe jakości specjalistów
uzyskanych niezależnymi prunerami, ponad predyktory bazowe, na nowych targetach.
To wynik ukierunkowanego wyszukania, nie dowód nieistnienia wcześniejszej pracy.
Samo dołożenie kilku granularności albo zebranie większej liczby modeli nie wystarczy.

## Zakres i sposób weryfikacji

- Punkt wyjścia: 41 pozycji we wcześniejszym audycie; nie wszystkie ponownie oceniono.
- Dziennik obejmuje 37 źródeł o różnych poziomach weryfikacji, nie 37 pełnych recenzji.
- Szukanie: kombinacje localization/modularity/circuits z compression/pruning,
  prediction/compressibility i task specialization; także wyszukiwanie tytułów.
- Źródła dowodowe: teksty autorów w arXiv, ACL/PMLR, repo autorów oraz jawnie
  oznaczone zindeksowane fragmenty pierwotnych PDF. Agregatory nie są dowodem.
- Uzupełnienie przez bibliografie: related work TaCQ, TASP i DiscoGP; dodatkowo
  ukierunkowane szukanie tytułów/cytowań i sprawdzenie starszych fundamentów.
  Nie wykonano pełnego forward citation search ani systematycznej deduplikacji baz.
- Czytano wskazane fragmenty metod, wyników i ograniczeń. Nie przeprowadzono
  reprodukcji, audytu całych repo autorów ani recenzji wszystkich dowodów.
- Dziennik zapytań i poziom dostępu: `literature_audit_20261011_sources.json`.

„Nie znaleziono testu X” poniżej dotyczy sprawdzonych fragmentów; nie oznacza
udowodnionej nieobecności X w całej pracy. Status konferencyjny/preprint nie jest
miarą poprawności wyników. Wyniki liczbowe są raportami autorów, nie naszą reprodukcją.

## Najbliższe precedensy

| Praca i dowód | Co faktycznie mierzy / kompresuje | Co zajmuje i co pozostaje do odróżnienia |
|---|---|---|
| [TASP](https://arxiv.org/html/2609.29499v1), §3.1–3.3, Appendix R/S | Deskryptory spektralne i salience przewidują ablation damage na rozłącznych modułach. Appendix R testuje bramkę diagnostyczną na sześciu nowych checkpointach; S obejmuje głowy i grupy SwiGLU. | **Najsilniejszy precedens predykcyjny.** Te same predykcje budują maski TASP; nie znaleziono tu testu frontierów innych prunerów na nowych targetach. Koszt główny: aktywne FLOPs i latency, nie minimalna fizyczna wielkość osobnego specjalisty. |
| [TaCQ](https://arxiv.org/html/2504.07389v2), §3, §4.2, Appendix H | Gradient, rozmiar wagi i symulowana zmiana po kwantyzacji wskazują wagi chronione w 16 bitach; resztę kwantyzuje GPTQ. Porównuje kalibrację ogólną i zadaniową oraz transfer. | Interpretability/localization już służy kompresji; „circuit-aware compression” jest zajęte. To konstrukcja metody kwantyzacji, nie znaleziony test miar modularności przewidujących krzywe niezależnego prunera. |
| [TAQ / You Had One Job](https://arxiv.org/html/2511.06516v4), §4.2–4.3, §5.2; v4, 28.06.2026 | Entropia reprezentacji i wariancja aktywacji ustalają precyzję warstw. TAQ-O mierzy spadek jakości po kwantyzacji pojedynczej warstwy. Autorzy raportują 87.5% identycznych przydziałów między zadaniami. | Representation-guided mixed precision i lokalna diagnoza wrażliwości są zajęte. Potrzebna kontrola wspólnej struktury warstw; nie zakładać dużych różnic zadaniowych. TAQ-O nie jest globalnym optimum kompresji. |
| [CGC](https://arxiv.org/pdf/2603.16440), §3.4–3.5, §5.1–5.5, §6 | Gęstość cech SAE steruje budżetem głów. Teoria zakłada związek aktywacji z redundancją geometryczną. Badanie GPT-2 Medium zawiera singleton ablations. | **Już testuje związek deskryptora z wrażliwością.** Density–ablation: Spearman −0.077, p=0.13; CGC-L ma gorsze PPL od uniform przy 50% retencji. Brak potwierdzenia w tym ustawieniu; nie dowód ogólnego braku związku. |
| [SubspacePath](https://proceedings.mlr.press/v306/zhiren26a.html) + [repo autorów](https://github.com/GongZhiren/SubspacePathPruner), README PSP | Probes na residual stream, dopasowanie zapisów głów do osi domen, backbone whitelist i scenariuszowa maska pod budżetem. Repo udostępnia artefakty dla czterech modeli. | Reprezentacje → specjalistyczna ścieżka i krzywe budżetów są zajęte. Wynik inferencyjny nie jest sam w sobie pomiarem konieczności. PDF niedostępny; README nie zastępuje oceny pełnych eksperymentów artykułu. |
| [LLM-Sieve](https://arxiv.org/html/2505.18350v2), §4.2, §5.2 | Rekonstrukcja low-rank i wyszukiwanie per-matrix budżetów względem jakości zadania; pruning ujawnia bottlenecki. | Minimalizacja rozmiaru przy tolerancji jakości i różnice między zadaniami są zajęte. Diagnoza organizacji wynika z optymalizacji; nie znaleziono niezależnego prognozowania jej wyników na nowych targetach. |
| [Decodable But Not Detachable](https://arxiv.org/html/2608.10214v1), §3, §4.6, §6 | Selektywność FFN, maskowanie i damage matrices; także INT4 fake quantization grup down-projection po 128 kanałów. | **Bada też kompresję grupową.** Przeplatanie neuronów uniemożliwia w ich ustawieniu wybór całkowicie core-only grup. Nie wolno odróżniać naszego pomysłu twierdzeniem „oni nie badają kompresji/grup”. Ich wynik nie ustanawia uniwersalnego prawa o pretrainingu. |
| [DiscoGP](https://aclanthology.org/2025.emnlp-main.446.pdf), §2–3, Table 2 | Wspólna optymalizacja masek wag i krawędzi; zero ablation sprawdza zdolność zachowanego fragmentu do samodzielnego działania. | Odkrywanie małych modułów wystarczających jest zajęte. Ta sama optymalizacja identyfikuje i przycina moduł; nie znaleziono niezależnej prognozy kompresowalności. Liczba aktywnych wag/krawędzi nie dowodzi oszczędności sprzętowej. |
| [AlphaPruning](https://arxiv.org/html/2410.10912), §4, Table 1 | Spektralna miara warstw ustala alokację sparsity; w obrębie warstw używa Magnitude, Wandy lub SparseGPT. | Struktura wag → budżety dla wielu prunerów jest zajęta. Nie zastępować własnego pytania nowym scorem alokacji; spektralne proxy to ważny punkt odniesienia dla funkcjonalnych miar. |
| [Perturbation–Absorption](https://arxiv.org/html/2606.15161), §3.2/3.5, §5.3 | Kontrolowane perturbacje i propagacja driftu; współczynnik absorpcji koryguje alokację OWL/AlphaPruning. Kontrola losowa zachowuje budżet. | Lokalna ważność nie opisuje całej odporności na pruning. Kompensacja w dalszych warstwach jest konkretną konkurencyjną hipotezą; nie znaleziono meta-predykcji specialist frontierów. |
| [ShadowLLM](https://aclanthology.org/2024.emnlp-main.1068.pdf), §3–6 | Predictor contextual sparsity uczy się per-input ważności głów/neuronów; porównuje activation i gradient criteria. | Przewidywanie ważności i użycie go do przycinania jest zajęte. Jednostka predykcji to kontekst, nie kompresowalność nowego targetu przez inną metodę. |
| [SEAP](https://ojs.aaai.org/index.php/AAAI/article/view/40463/44424), Method/Overview i Activation Patterns Modeling | Task-aware statystyki aktywacji, ocena retencji warstw i router wybierający maski. Deklaruje cross-task transfer. | Activation signatures → specjalistyczne ścieżki są zajęte. Odczytano opis konstrukcji; szczegółowego protokołu transferu nie re-audytowano. Sam transfer maski nie jest testem dodatkowej wartości predykcyjnej organizacji dla obcego prunera. |

CGC: mała korelacja rankowa z Wandą **nie dowodzi statystycznej niezależności**,
a proof sketch z założeniami geometrycznymi nie jest prawem o każdym LLM.
TASP: sześć checkpointów to mała próba; test jest ważnym precedensem, ale nie
uniwersalną gwarancją trafności bramki. Nie nazywać tych metod „cyrkularnymi” tylko
dlatego, że ich diagnoza konstruuje maskę: rozwiązują inne, poprawne pytanie badawcze.

## Pozostałe sprawdzone linie

| Źródło i odczytane miejsce | Znaczenie dla projektu |
|---|---|
| [D-Pruner](https://aclanthology.org/2024.findings-naacl.91.pdf), §3.3/4 | Domain/general importance, gradient/Fisher i specjaliści healthcare/legal. Zajmuje domain pruning; inny zakres niż miara organizacji prognozująca wynik obcej metody. |
| [Cus-Prun](https://aclanthology.org/2025.findings-acl.1201.pdf), §2 i wyniki scenariuszy | Specjalizacja w wymiarach język/domena/zadanie jest już porównywana w jednej pracy. Samo poszerzenie osi targetów nie zapewnia nowości. Szczegółów całego algorytmu nie zrecenzowano. |
| [TrimLLM](https://aclanthology.org/2025.acl-long.33.pdf), §3.3 i Table 3/7 | Layer dropping wybierany kalibracyjną ablacją, activation tie-breaker i fine-tuning. Zajmuje związek konieczności warstw ze specjalizacją; odróżnić model pretrained od odzyskiwania jakości treningiem. |
| [MixCal](https://aclanthology.org/2026.eacl-long.347.pdf), §5–6.1 | Wyniki na niewidzianych zadaniach i ablation data-mixture versus MixCal. Kalibracja domenowa jest istotnym confounderem i baseline'em; nie jest dowodem modularności. |
| [FineScope](https://arxiv.org/html/2505.00624), §3.1.3–3.2 | SAE wybiera dane domenowe; gradient-based LLM-Pruner przycina struktury, a teacher-guided distillation odzyskuje jakość. Interpretability-guided data selection plus domain compression jest zajęte. SAE nie identyfikuje tu bezpośrednio samowystarczalnego podmodelu. |
| [Capability Localization](https://arxiv.org/html/2502.20992), §4–5.1 | Kontrastowe prompty i interwencje na FFN; CNL agreguje integrated-gradient attribution na wyjściach całego zbioru. Lokalizacja wspólnych neuronów zdolności jest zajęta. Odczytane fragmenty nie ustanawiają prognozy krzywych niezależnego prunera; wyników §5.2 nie re-audytowano. |
| [KOFF](https://arxiv.org/html/2605.29075), §3.1, §4.4–4.5 | Wspólnie uczy maskę i pamięci LoRA/KV przy zamrożonych wagach bazowych. Module swaps oraz cross-reference z LAPE badają specjalizację. Zajmuje shared backbone + specialist memories; organizacja jest tu uczonym wynikiem prune-and-recover, nie wyłącznie pomiarem nietkniętego modelu. |
| [Configurable Foundation Models](https://arxiv.org/html/2409.02877), §4.1, §4.3–4.4 | Starszy szeroki framework modularnych „bricks” zawiera activation-based functionality scores i maskowanie neuronów. Taksonomia, overlap i próby interwencyjne nie są nowe. Sparsity per-token nie oznacza fizycznie małego statycznego specjalisty. |
| [Emergent Modularity](https://arxiv.org/html/2305.18390), §5 i Table 2 | Już w 2023 bada perturbacje funkcjonalnych ekspertów i ograniczenie routingu do nich z fine-tuningiem GLUE. To mocniejszy precedens niż sama obserwacja skupisk. Nie mylić jakości po adaptacji z zachowaniem jakości nietkniętego pretrained modelu. |
| [Are Neural Nets Modular?](https://people.idsia.ch/~csordas/are_neural_networks_modular.pdf), wstęp autora, wersja WHI 2020 | Zamraża wagi i uczy binarne maski dla zmienionych zadań; rozróżnia specjalizację i reuse. Potrzebny fundament operacyjnej modularności. Odczytany PDF jest wcześniejszą wersją, nie pełną re-recenzją wersji ICLR 2021. |
| [Impact of Calibration Data](https://aclanthology.org/2024.acl-long.544.pdf), §3 + oficjalny abstrakt ACL 2024 | Projekt porównania obejmuje 1800 skompresowanych modeli, cztery metody, dziewięć modeli i powtarzane pule kalibracyjne. Różnice downstream mogą wynikać z danych kalibracyjnych; liczba modeli w tabeli nie zastępuje niezależnych targetów naszego testu. |
| [Beware of Calibration Data](https://proceedings.iclr.cc/paper_files/paper/2025/file/2ede933e10afa991a10b6f36b6522129-Paper-Conference.pdf), §3.3–3.4, Table 1; ICLR 2025 | W badanym DCLM-7B/Wanda/DSnoT większa pula stabilizuje wyniki, lecz nie zamyka różnic między źródłami. Znajomość treningowego DCLM daje kontrolę podobieństwa. Nie przenosić tego jako uniwersalnego prawa na nieznany pretraining innych modeli. |
| [UniComp](https://arxiv.org/abs/2602.09130v6), abstrakt v6, 28.08.2026 | Ujednolicony benchmark pruning/quantization/distillation i task-specific calibration. Ogólny benchmark wielu zdolności też jest zajęty. Abstrakt nie pozwala ocenić pełnego związku jego wyników z miarami organizacji. |
| [OCP](https://aclanthology.org/2026.acl-long.276/), oficjalny abstrakt ACL 2026 | Input-adaptive probing i alokacja sparsity według outlier statistics. Dynamiczne wyznaczanie kanałów jest istniejącym kierunkiem; zakres odróżnić od statycznego specialist frontier. Pełnej metodologii nie oceniano. |
| [Task-aware Pruning](https://arxiv.org/html/2510.22489), §II-B/IV-D | Normy aktywacji ogólnych/docelowych dzielą kanały na shared/general/task. Taki podział nie jest nowy ani automatycznie przyczynowy. |
| [EfficientXpert](https://arxiv.org/html/2511.19935), §3–4 | Wrażliwość domen po LoRA i propagation-aware mask/recovery. Utrzymanie jakości po fine-tuningu nie jest tym samym estimandem co retencja zdolności oryginalnego pretrained LLM. |
| [Translation Specialists](https://arxiv.org/html/2605.28042), §4.2/5.2/6 | Routing mass i routing divergence przydzielają budżet ekspertów MoE. Krzywe przenoszą się na inne domeny tłumaczenia; kontrole odwrócone badają użyteczność doboru. Zachowane parametry poza MoE nadal współtworzą specjalistę. |
| [Localization → Editing](https://arxiv.org/html/2301.04213), §4.1–4.3/5.3 | Causal tracing wnosi mało ponad wybór warstwy przy prognozowaniu skuteczności edycji. Bezpośredni wzorzec porównania incremental prediction; wynik nie przesądza kompresji. |
| [Localization → Unlearning](https://arxiv.org/html/2505.16252), §3–4, Appendix E | Kontrolowane oracle/random regiony i alternatywne cele/jednostki. Raport marginalnych różnic nie dowodzi równoważności; motywuje losowe kontrole i z góry ustalony meaningful effect. |
| [Cross-Lingual Generalization](https://aclanthology.org/2025.acl-long.661.pdf), abstrakt i fragment analizy checkpointów | „Compression” oznacza powstawanie wspólnych reprezentacji w treningu, a nie zmniejszanie wag. Użyteczna dla organizacji języków; nie mylić osi kosztu. |
| [Asymmetric Harms](https://arxiv.org/html/2608.19670v1), abstrakt/definicje i Appendix A headings | Autorzy raportują utratę względnej retencji wiedzy head; porównania zależą od datasetu i miary. Popularity proxy nie jest bezpośrednim pomiarem częstości w nieznanym pretraining corpus. |
| [Compressed Sensing](https://proceedings.mlr.press/v306/bair26a.html), oficjalny abstrakt | Nieliczne głowy konieczne dla zdolności. Nie wynika z tego, że pozostawienie samych tych głów wystarczy do działania modelu. Pełnej metodologii nie re-audytowano. |

## Trafienia pozornie identyczne i ograniczenia dostępu

- [Connectome](https://openreview.net/pdf/9a49e30eea369ea5330cef29c63a69773fa3409a.pdf): zindeksowany pierwotny fragment metod, linie 549–557, definiuje ratio jako
  liczba węzłów grafu / liczba superwęzłów po Laplacian renormalization. To kompresja
  grafu cech. Odczytany fragment nie jest wynikiem przycinania wag specjalisty;
  pełny PDF blokuje browser challenge, więc pozostały zakres zostaje niezweryfikowany.
- [MARC](https://arxiv.org/abs/2604.18146): trafienie abstraktowe dotyczące kompresji
  reprezentacji dla rekomendacji; nie traktowano go jako dowodu redukcji wag LLM.
- [Sapling](https://openreview.net/pdf?id=vzvCaYFTLq): zindeksowany fragment wersji
  under review opisuje warstwową specjalizację z treningiem. Nie ustalono relacji
  bibliograficznej do TrimLLM; nie liczyć jako niezależnego nowego precedensu.
- SubspacePath: PMLR PDF i OpenReview były niedostępne; analizowano oficjalny
  abstrakt i opis autorów w repo. Dalszy dostęp może zmienić szczegółowy werdykt.
- Szukanie prac powiązanych dało też agregatory i niesprawdzone „hypotheses”.
  Nie awansowano ich do dowodów. Weryfikowano odnalezione źródła pierwotne;
  brak trafienia po tytule nie dowodzi braku cytujących prac ani nowości hipotezy.

## Operacjonalizacja naszego możliwego wkładu

**Czy tanie, przyczynowe miary organizacji w nietkniętym modelu dodają wartość
predykcyjną dla niezależnie otrzymanych krzywych przycinania na nowych targetach?**

Jednostka analizy: `(model, target, pruner)`, z osobnymi cechami dla każdej
granularności. Kilka progów sparsity i calibration seeds w tej samej jednostce
to powtarzane pomiary, nie niezależne targety. Predykcja powinna obejmować krzywą
lub z góry określony wynik; nie wybierać najlepszego score/epsilon po obejrzeniu testu.

Wkład wymaga przynajmniej:
1. Discovery measurements ustalone przed poznaniem frontierów; nie sterują
   niezależnym prunerem. Dodatkowo rozłączne pule organization/calibration wewnątrz
   discovery ograniczą wspólny szum estymacji; ich rozdział trzeba zapisać w protokole.
2. Porównania baseline-only z baseline+organization, na held-out targetach;
   oddzielnego testu transferu między modelami i prunerami, z kosztem pomiaru.
3. Kontroli baseline quality/chance level, global pruning sensitivity, rozmiaru
   próbki, calibration tokens/seed, typu zadania i dostępnych proxy popularności.
   Spektralne miary i kompensacja to konkurencyjne wyjaśnienia, nie automatyczne dodatki.
   TAQ dodatkowo motywuje baseline oparty na wspólnej wrażliwości warstw między zadaniami.
   Prace o kalibracji wymagają powtórzeń po jej źródłach/seedach, przy równych budżetach.
4. Jawnej osi kosztu: nonzero weights, fizyczne bytes i FLOPs nie są wymienne.
   Nasza obecna Wanda mierzy retencję przy zerowaniu wag w gęstym storage; nie dowodzi
   zmniejszenia VRAM. Core wymaga osobnego sensownego benchmarku; obecne core-values
   na jednej osi targetów nie zapewniają ogólnego core językowo-rozumującego.
5. Niepewności over examples/calibration seeds/targets i prerejestrowanego minimalnego
   useful effect. Dobrej mocy nie zastąpi liczba masek ani duża liczba wierszy wyników.

Hipoteza dodatnia nie jest konieczna do wartościowego badania, ale nieistotne p-value
nie dowodzi braku praktycznie ważnego związku. Wynik negatywny wymaga precyzyjnych
przedziałów albo testu równoważności względem ustalonej granicy, a nie samego braku istotności.

Miara konieczności nie gwarantuje wystarczalności: mały specyficzny „gateway” może
być konieczny, podczas gdy zdolność nadal wymaga dużego wspólnego core. To logiczny
kontrprzykład prostego „bardziej skupione = bardziej kompresowalne”, nie wynik repo.
Stąd **prediction**, nie automatyczne mechanistyczne wyjaśnienie ani uniwersalne prawo.

Empiryczna krzywa konkretnego prunera jest dolnym ograniczeniem optymalnej jakości
przy danym koszcie; znaleziony minimalny koszt jest górnym ograniczeniem idealnego
minimalnego kosztu. Na rzadkiej siatce budżetów nie obserwujemy dokładnego optimum.

## Decyzja praktyczna

**Główny cel publikacyjny: ICML 2027.** Obecny kierunek ma potencjał, ale sam
odmienny protokół, korelacja albo framework nie uzasadniają wystarczającego wkładu.
[Instrukcje recenzentów ICML 2026](https://icml.cc/Conferences/2026/ReviewerInstructions)
rozróżniają poprawność, oryginalność i znaczenie; dopuszczają nowy wgląd wynikający
z badania istniejących metod. To punkt odniesienia, nie zweryfikowany regulamin 2027.

Nasza ocena progu skalowania: pilot powinien ujawnić albo użyteczną predykcję na
nowych targetach ponad tanie baseline'y, albo powtarzalny kontrprzykład z dowodami
wyjaśniającymi, dlaczego lokalizacja zawodzi. Sam brak istotności nie spełnia tego progu.
Dla mocnego paperu trzeba sprawdzić zakres wyniku między modelami/prunerami oraz
pokazać jego konsekwencję: np. trafniejszy wybór budżetu specjalisty albo rozróżnienie
koniecznego lokalnego komponentu i dużego wspólnego core. To propozycja strategii,
nie obietnica akceptacji ani formalne wymaganie liczby modeli.

**Kontynuować mały, falsyfikowalny pilot; nie deklarować potwierdzonej nowości ICML.**
Kod narzędzi pozostaje przydatny. Następny etap to zamrożenie narrow claim i protokołu,
domknięcie danych/evaluatorów fazy 1, a potem walidacja na pretrained modelu i pilot.
Pilot ma pokazać incremental prediction i przedziały; dopiero wtedy uzasadniać
skalowanie. Dostęp do pełnego SubspacePath i odświeżenie forward citations przed
submission pozostają konkretnymi lukami audytu. Koszt runów zmierzyć na serwerze;
sam iloczyn targety × budżety × modele nie daje rzetelnego czasu obliczeń.
