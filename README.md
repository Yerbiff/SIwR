# SIwR

## Struktura

1. **parse_labels**: Funkcja do parsowania plików z etykietami ramek.
2. **calculate_histogram**: Funkcja do obliczania histogramu dla obszaru zainteresowania (ROI).
3. **calculate_distance**: Funkcja do obliczania odległości między dwoma ramkami (bounding boxes).
4. **build_factor_graph**: Funkcja do budowania grafu czynników.
5. **perform_inference**: Funkcja do przeprowadzania wnioskowania przy użyciu propagacji wiarygodności.
6. **accuracy_metric**: Funkcja do obliczania dokładności wnioskowania.
7. **main**: Główna funkcja programu.

## Rozwiązania

Rozwiązanie polega na konstrukcji grafu czynników (FactorGraph) oraz wykorzystaniu propagacji wiarygodności (BeliefPropagation) do przeprowadzenia wnioskowania. Dla każdej pary ramek tworzony jest oddzielny graf.

### Graf

Węzły zmiennych (X_i) reprezentują indeksy ramek w drugiej klatce, które muszą zostać dopasowane. Liczba węzłów odpowiada liczbie ramek w drugiej klatce. Po zakończeniu propagacji wiarygodności, węzły te przyjmują najbardziej prawdopodobne wartości indeksów ramek. Do zapobiegania sytuacji, w której klatka uzyskuje wiecej niż jedno dopasowanie, dodany jest kolejny czynnik który także łączy każde pozostałe węzły ze sobą. Macierz wartości tych czynników zawiera jedynki we wszystkich komórkach poza główną przekątną, gdzie znajdują się zera. Wartość na pozycji [0, 0] wynosi 1, co pozwala na sytuację, w której żaden węzeł nie jest przypisany do ramki.
Najważniejsze czynniki to te porównujące histogramy dwóch 'sąsiadujących' klatek. Czynniki te zmuszają węzły do przyjęcia indeksu ramki o najwyższym podobieństwie histogramów. Jeśli żadne podobieństwo nie przekracza określonego progu, węzeł przyjmuje wartość 0.
Kolejnym istotnym elementem są czynniki mierzące odległości, które obliczają odległości między każdą ramką z pierwszej klatki a każdą ramką z drugiej klatki. Czynniki te zmuszają węzeł do przyjęcia indeksu najbliższej ramki. Jeśli żadna ramka nie znajduje się bliżej niż określona maksymalna odległość, węzeł przyjmuje wartość 0. Im bliżej siebie znajdują się ramki w różnych klatkach, tym mniejsza jest ich odległość i tym wyższa wartość czynnika. Gdy czynniki przyjmują wartość 0 na potrzeby programu zamieniane są na wartość -1 by uzyskiwać odpowiednie wyniki.






