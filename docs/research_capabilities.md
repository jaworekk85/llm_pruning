# Co można zbadać z wykorzystaniem tego repo?

Stan na 15 października 2026 r. Przewodnik po możliwościach i planowanych rozszerzeniach.
Aktualne priorytety ma [strategia badawcza](research_strategy.md), a kolejność
uruchomień na nowym serwerze — [instrukcja serwerowa](server_handoff.md).

## 1. Do czego to służy?

Repo pozwala badać, **które części LLM wspierają określone zadania i co dzieje się
z tymi zadaniami po ingerencji w model**. Docelowo chcemy wykorzystać taką wiedzę
do budowania mniejszych modeli zachowujących wybrane umiejętności.

Można zadawać trzy rodzaje pytań:

- **Organizacja:** czy różne dziedziny, języki lub umiejętności korzystają z podobnych
  komponentów? Czy ich wsparcie jest skupione, rozproszone i stabilne między pomiarami?
- **Przyczynowość:** czy wyłączenie wskazanych komponentów rzeczywiście szkodzi danej
  umiejętności? Czy szkodzi jej bardziej niż innym oraz bardziej niż losowa ingerencja?
- **Kompresja:** jak mocno można przyciąć model, zachowując wybraną jakość? Czy mapa
  jego organizacji pomaga przewidzieć lub poprawić ten wynik?

To szerszy program niż obecny pilot ICML. Możemy zmienić hipotezę lub zestaw zadań
i nadal wykorzystać wspólne narzędzia. „Mapa wiedzy” oznacza tutaj pomiary zależne
od zadań i interwencji; nie mamy katalogu wskazującego lokalizację każdego faktu.

## 2. Co jest gotowe, a co dopiero planujemy?

**Gotowe w kodzie nie oznacza potwierdzone badawczo.** Ostatni zestaw testów
technicznych obejmuje 282 testy offline na małych, losowo inicjalizowanych modelach.
Nie mamy jeszcze naukowych wyników z docelowego modelu pretrained.

| Możliwość | Stan |
|---|---|
| Budowa i kontrola obecnych zadań kompozycyjnych | Gotowe; działa lokalnie bez GPU |
| Analiza zapisanych pomiarów i niepewności | Gotowe; bez GPU, jeśli istnieją odpowiednie wyniki |
| Pomiar aktywacji i wyłączanie głów uwagi / grup MLP w zadaniach wielokrotnego wyboru | Gotowe w kodzie; docelowe pomiary wymagają zasobów do inferencji |
| Przycinanie Wandą i porównanie jakości przy różnych budżetach wag | Gotowe w kodzie; wymaga uruchomienia na modelu pretrained |
| Test kompozycji przed i po przycinaniu, z kontrolami | Gotowe w kodzie; czeka na sprawdzenie kompetencji modelu na GPU |
| Przyczynowa mapa kompozycji i ochrona jej komponentów podczas kompresji | Wymaga rozszerzenia kodu; następny etap zależny od pilota |
| Pełne eksperymenty wielojęzyczne, strumieniowe i fizyczna kompresja | Kierunki rozwoju; brak kompletnych ścieżek eksperymentalnych |

Dane rozdzielamy na trzy części: **discovery** do wyboru komponentów i kalibracji,
**validation** do podejmowania decyzji o eksperymencie oraz **test** do końcowej
oceny po zamrożeniu tych decyzji. Tego samego przykładu nie używamy do wyboru
komponentów i końcowego potwierdzania ich znaczenia.

## 3. Konkretne eksperymenty dostępne obecnie

### A. Profil umiejętności modelu

**Pytanie:** z czym model sobie radzi i czy ma sens badać utratę tej umiejętności?

Na zbiorze pytań wielokrotnego wyboru można porównać jakość według etykiet dziedziny,
języka, umiejętności lub zadania. Przykład: zestawić kilka dziedzin z MMLU,
zachowując ten sam sposób oceniania. Nowe języki wymagają przygotowania danych.

Wynik to jakość i decyzje dla poszczególnych przykładów oraz grup. To punkt
odniesienia dla kolejnych interwencji; słaby wynik bazowy utrudnia interpretację.
Kod: `experiments/icml2027/run_target_baseline.py`.

### B. Mapa aktywności i podobieństwa między zadaniami

**Pytanie:** które głowy uwagi i grupy neuronów MLP wyróżniają się dla danego celu?

Można zebrać aktywacje, policzyć selektywność względem innych celów, koncentrację
wyników oraz nakładanie się komponentów między celami. Powtórzone pomiary pozwalają
sprawdzić stabilność rankingu. Przykład: czy dwie dziedziny mają wspólne komponenty
o wysokiej selektywności, czy każda wyróżnia inny zestaw?

To **opis aktywności**, a nie dowód przechowywania wiedzy lub konieczności komponentu.
Obecna uporządkowana ścieżka obsługuje pytania wielokrotnego wyboru.
Narzędzia: `collect_component_activations.py`, `analyze_component_organization.py`
w `experiments/`. Szczegóły: [pomiary organizacji](pruning_and_organization.md).

### C. Test przyczynowy: co się stanie po wyłączeniu komponentów?

**Pytanie:** czy wskazane komponenty są rzeczywiście ważne dla wykonania zadania?

Można wyłączyć pojedynczą głowę / grupę MLP albo zestaw wybrany na danych
odkrywczych, a następnie zmierzyć szkody na osobnych danych. Porównania obejmują
losowe maski oraz maski dopasowane pod względem warstw i strukturalnego kosztu.
Przykład: czy wyłączenie komponentów wybranych dla biologii pogarsza biologię
bardziej niż inne dziedziny i bardziej niż porównywalne losowe wyłączenie?

Wynik mówi o skutku konkretnej interwencji. Efekty pojedynczych komponentów nie
muszą się sumować; model może mieć zastępcze ścieżki obliczeń. Dopasowanie kosztu
jest obecnie przybliżeniem strukturalnym, nie pomiarem czasu GPU.
Narzędzia: `probe_component_importance.py`, `evaluate_component_masks.py`.
Protokół: [ewaluacja przyczynowa](component_causal_evaluation.md).

### D. Specjalista: ile wag można wyzerować bez dużej utraty jakości?

**Pytanie:** jak zmienia się jakość wybranego zadania przy coraz mocniejszym przycinaniu?

Wanda jest metodą odniesienia: wybiera wagi na podstawie ich wartości i aktywacji
na danych kalibracyjnych. Można kalibrować ją na wybranym celu, przejść przez kilka
budżetów pozostawionych wag i sprawdzić jakość zarówno tego celu, jak i pozostałych.
Przykład: czy model przycinany pod jedną dziedzinę zachowuje jej jakość dłużej niż
jakość innych dziedzin? Gdzie zaczyna się nieakceptowalny spadek?

Wynik to zaobserwowana krzywa jakości względem liczby niezerowych wag. Można ją
zestawić z mapami z B–C, ale analiza, czy organizacja **przewiduje** kompresowalność,
wymaga jeszcze zaprojektowania porównania wielu niezależnych celów i kontroli.
Wanda nie korzysta obecnie z naszych map komponentów.

Wyzerowane wagi nadal zajmują miejsce w gęstych tensorach: ten eksperyment nie
dowodzi zmniejszenia pliku, zużycia VRAM ani przyspieszenia. Implementacja obsługuje
określone modele Llama/Qwen2 z pełną uwagą na jednym urządzeniu.
Narzędzie: `run_compression_frontier.py`; [możliwości i ograniczenia](pruning_and_organization.md).

### E. Kompozycja: czy model zachowuje części umiejętności, ale traci ich połączenie?

**Pytanie:** czy przycinanie szkodzi zadaniu A→B bardziej, niż wynikałoby z uszkodzenia A i B?

To obecny pierwszy pilot dla ICML 2027. Mamy dwie rodziny zadań:

- **Odczyt → odczyt:** z kontekstu ustal kod obiektu, a potem wartość przypisaną do kodu.
- **Odczyt → dodawanie:** z kontekstu odczytaj liczbę, a potem dodaj zadaną wartość.

Dla tych samych przypadków mierzymy A, B, całe A→B, wykonanie dwuetapowe
z rzeczywistym wynikiem A oraz B z podanym poprawnym wynikiem pośrednim.
Kontrole zmieniają informację istotną dla odpowiedzi, dystraktory, kolejność
i nazwy obiektów. Pozwala to sprawdzić korzystanie z właściwych danych i odróżniać
błąd części składowej, propagację błędu oraz trudność połączenia kroków.

Te same zadania można ocenić na modelu gęstym i po Wandzie. Warianty dwuetapowe
i z podpowiedzią służą diagnozie; mają inny koszt lub dodatkową informację.
Powiązane warianty jednego przypadku nie są niezależnymi obserwacjami.

**Granica obecnej możliwości:** fakty znajdują się w kontekście, więc badamy ich
wykorzystanie, a nie przechowywanie w wagach. Nie ma jeszcze adaptera łączącego
ten ewaluator generacyjny z przyczynowym mapowaniem z punktu C ani celowanej
ochrony komponentów. Samo wykazanie spadku kompozycji nie wystarczy do tezy o nowości.
Narzędzie: `experiments/icml2027/run_composition_screen.py`;
[projekt zadań](composition_benchmark_design.md), [protokół](projects/icml2027.md).

### F. Jak wiarygodny jest wynik i ile losowych masek potrzeba?

**Pytanie:** czy obserwowana różnica jest stabilna względem przykładów i losowych masek?

Dla zapisanych eksperymentów przyczynowych można lokalnie obliczać przedziały
niepewności i szacować liczbę masek potrzebną do zadanej precyzji. Liczba losowych
masek jest parametryzowana; domyślna mała liczba nie zapewnia istotności ani mocy.
To analiza istniejących wyników, bez ponownego ładowania modelu.

Kompozycja ma osobne grupowanie przypadków i osobne planowanie liczebności.
Nie należy przenosić założeń statystycznych między tymi ścieżkami automatycznie.
Narzędzie: `analyze_mask_uncertainty.py`;
[niepewność masek](mask_uncertainty.md), [plan walidacji kompozycji](composition_validation_plan.md).

## 4. Co możemy robić teraz lokalnie?

Bez serwera można budować i audytować dane, sprawdzać izolację zbiorów, przygotować
konfiguracje i wykonać kontrole bez ładowania modelu. Można także analizować istniejące
wyniki, planować precyzję pomiarów oraz rozwijać i testować kolejne moduły na małych
modelach. To prace techniczne i metodologiczne; nie zastąpią pomiaru kompetentnego LLM.

**Do uruchomienia pierwszego pilota nie trzeba już dopisywać kolejnej infrastruktury.**
Najbardziej wartościowy następny krok to pomiar modelu gęstego na serwerze.
Nie potrzebujemy teraz rozbudowywać repo na zapas.

## 5. Przewidywalna kolejność rozwoju po uzyskaniu GPU

Kolejne kroki zależą od wyników, a nie od założenia, że hipoteza musi się potwierdzić:

1. **Sprawdzić kompetencję modelu.** Zacząć od małego testu modelu gęstego:
   8 niezależnych grup przypadków, łącznie 40 epizodów z kontrolami. Ocenić też
   parser, zachowanie wobec zmian danych i koszt wykonania. To test wykonalności.
2. **Rozszerzyć walidację i uruchomić małą serię prób przycinania.** Jeśli zadania działają,
   przejść do większej walidacji, sprawdzić precyzję oraz porównać wybrane budżety Wandy.
   Zamrozić potrzebne decyzje przed testem; obecne dane mają po 128 grup na podział.
3. **Jeśli pojawi się przekonujący efekt — zmapować jego wsparcie.** Dodać pomiary
   przyczynowe dla generowanych odpowiedzi. Sprawdzić, czy wsparcie A, B i A→B
   pokrywa się oraz co zmienia się po kompresji. To nowa implementacja.
4. **Przetestować użyteczność mapy.** Zaimplementować celowane zachowanie lub
   odtworzenie wsparcia i porównać z losową oraz inną dopasowaną ochroną przy tym
   samym budżecie. Oceniać również szkody dla innych zadań.
5. **Sprawdzić przenoszenie wyniku.** Drugi model, inna metoda kompresji i nowe
   kombinacje zadań. Dopiero całość pozwoli ocenić siłę konkretnego wkładu do ICML.

Jeśli pierwszy pilot zawiedzie, wynik pomoże poprawić benchmark lub pytanie.
Wspólne narzędzia pozostają użyteczne dla innych eksperymentów z punktów A–D.

## 6. Co można dobudować dla UCT i Waikato?

To propozycje do uzgodnienia z partnerami. Nie mamy jeszcze dedykowanych,
kompletnych implementacji tych projektów ani uzgodnionych terminów.

| Kierunek | Konkretny eksperyment | Co wykorzystamy / co trzeba dodać |
|---|---|---|
| UCT: organizacja języków | Porównać wspólne i specyficzne wsparcie kilku języków w porównywalnych zadaniach | Metadane, aktywacje, maski; potrzebne dane, audyt porównywalności i ocena kompetencji językowej |
| UCT: specjalista językowy | Kalibrować przycinanie na wybranym języku i mierzyć jakość tego języka oraz straty w innych | Wanda i ewaluacja; potrzebne uzgodnione benchmarki, osobny pomiar jakości modelowania języka i ewentualnej adaptacji |
| Waikato: dezaktualizacja specjalisty | Zmieniać w czasie zapotrzebowanie na dziedziny lub języki i mierzyć moment utraty przydatności modelu | Statyczne pomiary jakości; trzeba dodać strumień i ocenę zgodną z kolejnością napływu danych |
| Waikato: reakcja na zmianę | Porównać stałego specjalistę, model ogólny i zmianę pojemności / ponowną kalibrację | Potrzebny mechanizm decyzji, obsługa opóźnionych etykiet i rozliczenie kosztu adaptacji oraz modelu zapasowego |

## 7. Dalsze możliwości, które wymagają osobnego projektu

- **Wiedza zapisana w wagach:** zadania bez podawania faktów w kontekście,
  potwierdzenie znajomości faktów, odporność na parafrazy i ingerencje przyczynowe.
  Import faktów liczbowych jest zalążkiem danych; pełny protokół lokalizacji wiedzy
  parametrycznej wymaga pracy.
- **Wsparcie zastępcze i interakcje:** pomiar komponentu po uszkodzeniu innej części
  modelu; ważność w modelu nienaruszonym może nie przewidywać roli po kompresji.
  Obecne maski są podstawą, ale nie ma kompletnego eksperymentu tego rodzaju.
- **Fizycznie mniejszy lub szybszy model:** przycinanie strukturalne, kwantyzacja
  lub inny format wykonania, a następnie pomiar rozmiaru, VRAM, jakości i czasu.
  Wymaga osobnej implementacji i pomiarów sprzętowych.
- **Odzyskiwanie jakości:** dostrajanie, destylacja i porównanie z małym modelem
  trenowanym od zera. Nie mamy gotowej ścieżki tych porównań; trzeba uwzględnić
  dostęp do danych oraz koszt treningu i adaptacji.

Nie planujemy wdrażać wszystkich tych kierunków naraz. Dla obecnego ICML priorytetem
jest sekwencja z punktu 5; szersze możliwości są opcjami, nie obiecanymi wynikami.

## 8. Jakich zasobów to wymaga?

Budowanie danych, kontrole konfiguracji i analizy zapisanych wyników działają na CPU.
Pomiary modelu 7–8B planujemy na serwerze GPU; dokładne wymagania zależą od modelu,
precyzji, długości kontekstu i batcha. Wanda potrzebuje również RAM na kopie wag
i dane kalibracyjne. Pełne mapowanie może być kosztowne, bo kolejne interwencje
wymagają ponownego oceniania tych samych przykładów.

Najpierw mierzymy mały przebieg, a dopiero z niego szacujemy większą serię.
Obecne liczniki zasobów mają ograniczenia opisane w protokołach; nie stanowią pełnego
benchmarku szybkości i pamięci. Nie obsługujemy dowolnej architektury LLM, modeli
MoE, układów rozproszonych ani dowolnych modeli już skwantyzowanych.

Do uruchomień używaj [instrukcji serwerowej](server_handoff.md), do znalezienia kodu
[mapy modułów](code_map.md), a do sprawdzenia bieżącego kroku — [STATUS](../STATUS.md).
