import os
import itertools
import argparse

import numpy as np
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import cv2


def parse_labels(root_path, with_ground_truth=False):
    """
    Funkcja do parsowania plików z etykietami ramek.

    Parametry:
    - root_path (str): Ścieżka do głównego katalogu zbioru danych.
    - with_ground_truth (bool): Flaga określająca, czy należy użyć etykiet ground truth.

    Zwraca:
    - frames (list): Lista krotek zawierających informacje o ramkach.
    - per_frame_ground_truths (list): Lista etykiet ground truth dla każdej ramki.
    """
    # Określenie pliku do wczytania na podstawie flagi with_ground_truth
    target = 'bboxes.txt' if not with_ground_truth else 'bboxes_gt.txt'
    # Wczytanie linii z pliku
    with open(os.path.join(root_path, target), 'r') as f:
        lines = f.readlines()
    frames = []
    per_frame_ground_truths = []
    # Przetwarzanie każdej linii z pliku
    while len(lines) > 0:
        source_file = lines.pop(0).strip()
        bbox_n = int(lines.pop(0).strip())
        coordinates = []
        ground_truths = []
        for _ in range(bbox_n):
            if with_ground_truth:
                ground_truth, *coords = lines.pop(0).strip().split()
                ground_truths.append(ground_truth)
                coordinates.append(coords)
            else:
                coordinates.append(lines.pop(0).strip().split())
        coordinates = [[float(coord) for coord in coordinate] for coordinate in coordinates]
        frames.append((source_file, bbox_n, coordinates))
        if with_ground_truth:
            per_frame_ground_truths.append(ground_truths)
    return frames, per_frame_ground_truths


def calculate_histogram(roi):
    """
    Funkcja do obliczania histogramu dla obszaru zainteresowania (ROI).

    Parametry:
    - roi (np.ndarray): Obszar zainteresowania w formie macierzy Numpy.

    Zwraca:
    - hist_hsv_roi (np.ndarray): Histogram obrazu w przestrzeni barw HSV.
    """
    # Konwersja obrazu z BGR do HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ranges = [0, 180, 0, 256]
    # Ustawienia dokładności dla Hue i Saturation
    h_bins = 50
    s_bins = 60
    hist_size = [h_bins, s_bins]
    # Obliczanie histogramu
    hist_hsv_roi = cv2.calcHist([hsv_roi], [0, 1], None, hist_size, ranges, accumulate=False)
    # Normalizacja histogramu
    cv2.normalize(hist_hsv_roi, hist_hsv_roi, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist_hsv_roi


def calculate_distance(box1, box2):
    """
    Funkcja do obliczania odległości między dwoma bounding boxami.

    Parametry:
    - box1 (list): Lista zawierająca współrzędne pierwszego bounding boxa (x, y, width, height).
    - box2 (list): Lista zawierająca współrzędne drugiego bounding boxa (x, y, width, height).

    Zwraca:
    - distance (float): Odległość między środkami obu bounding boxów.
    """
    # Obliczenie środków bounding boxów
    box1_center = np.array([box1[0] + box1[2] / 2, box1[1] + box1[3] / 2])
    box2_center = np.array([box2[0] + box2[2] / 2, box2[1] + box2[3] / 2])
    # Obliczenie euklidesowej odległości między środkami
    distance = np.linalg.norm(box1_center - box2_center)
    return distance


def build_factor_graph(frame1_bboxes, frame2_bboxes, frame1_histograms, frame2_histograms, params):
    """
    Funkcja do budowania grafu czynników.

    Parametry:
    - frame1_bboxes (list): Lista zawierająca bounding boxy dla pierwszej klatki.
    - frame2_bboxes (list): Lista zawierająca bounding boxy dla drugiej klatki.
    - frame1_histograms (list): Lista histogramów dla pierwszej klatki.
    - frame2_histograms (list): Lista histogramów dla drugiej klatki.
    - params (dict): Słownik zawierający parametry konfiguracyjne.

    Zwraca:
    - G (FactorGraph): Graf czynników.
    """
    # Tworzenie grafu czynników
    G = FactorGraph()
    frame1_len = len(frame1_bboxes)
    frame2_len = len(frame2_bboxes)

    # Dodawanie węzłów zmiennych dla każdego bounding boxa w drugiej klatce
    for index in range(frame2_len):
        G.add_node(f'X_{index}')

    # Dodawanie czynników porównania histogramów
    for frame2_bbox_idx in range(frame2_len):
        factor_similarity_values = [params['prog_podobienstwa_hist']]
        for frame1_bbox_idx in range(frame1_len):
            # Porównanie histogramów
            similarity = cv2.compareHist(
                frame1_histograms[frame1_bbox_idx],
                frame2_histograms[frame2_bbox_idx],
                cv2.HISTCMP_CORREL)
            factor_similarity_values.append(similarity)
        df = DiscreteFactor(
            [f'X_{frame2_bbox_idx}'],
            [frame1_len + 1],
            [factor_similarity_values])
        G.add_factors(df)
        G.add_edge(f'X_{frame2_bbox_idx}', df)

    # Dodawanie czynników unikania duplikacji
    # Zapobiega wielokrotnemu dopasowaniu tej samej ramki ograniczającej.
    values = np.ones((frame1_len + 1, frame1_len + 1))
    np.fill_diagonal(values, 0)
    values[0, 0] = 1  # Dla przypadku, gdy nie jest wybrany żaden bounding box
    for i, j in itertools.combinations(range(frame2_len), 2):
        df = DiscreteFactor(
            [f'X_{i}', f'X_{j}'],
            [frame1_len + 1, frame1_len + 1],
            values)
        G.add_factors(df)
        G.add_edge(f'X_{i}', df)
        G.add_edge(f'X_{j}', df)

    # Dodawanie czynników odległości
    for frame2_bbox_idx in range(frame2_len):
        inv_distances = [1 / params['prawdopodobna_max_odlegloosc']]
        for frame1_bbox_idx in range(frame1_len):
            distance = calculate_distance(frame1_bboxes[frame1_bbox_idx], frame2_bboxes[frame2_bbox_idx])
            inv_distance = 1 / distance if distance != 0 else 1 / 0.00001
            inv_distances.append(inv_distance)
        df = DiscreteFactor(
            [f'X_{frame2_bbox_idx}'],
            [frame1_len + 1],
            [inv_distances])
        G.add_factors(df)
        G.add_edge(f'X_{frame2_bbox_idx}', df)

    return G


def perform_inference(G):
    """
    Funkcja do przeprowadzania wnioskowania przy użyciu propagacji wiarygodności.

    Parametry:
    - G (FactorGraph): Graf czynników.

    Zwraca:
    - matching (dict): Dopasowanie (mapowanie) między zmiennymi a ich wartościami.
    """
    # Inicjalizacja propagacji wiarygodności
    belief_propagation = BeliefPropagation(G)
    # Kalibracja
    belief_propagation.calibrate()
    # Wnioskowanie (mapowanie) wartości zmiennych
    matching = belief_propagation.map_query(G.get_variable_nodes())
    # Korekcja indeksów, aby zaczynały się od -1
    matching.update((key, value - 1) for key, value in matching.items())
    return dict(sorted(matching.items()))


def accuracy_metric(matching, ground_truths):
    """
    Funkcja do obliczania dokładności wnioskowania.

    Parametry:
    - matching (dict): Dopasowanie (mapowanie) między zmiennymi a ich wartościami.
    - ground_truths (list): Lista etykiet ground truth.

    Zwraca:
    - correct (int): Liczba poprawnych dopasowań.
    - incorrect (int): Liczba niepoprawnych dopasowań.
    """
    correct = sum([1 for match, ground_truth in zip(matching, ground_truths) if match == ground_truth])
    incorrect = len(matching) - correct
    return correct, incorrect


def main(dataset_root, with_ground_truth=False, verbose=False):
    """
    Główna funkcja programu.

    Parametry:
    - dataset_root (str): Ścieżka do głównego katalogu zbioru danych.
    - with_ground_truth (bool): Flaga określająca, czy należy użyć etykiet ground truth.
    - verbose (bool): Flaga określająca, czy wyświetlać szczegółowe informacje.
    """
    frames, ground_truths = parse_labels(os.path.join(dataset_root), with_ground_truth)

    correct_cumulative = 0
    incorrect_cumulative = 0

    params = {
        'prawdopodobna_max_odlegloosc': 120,
        'prog_podobienstwa_hist': 0.25
    }

    if verbose:
        print(f"Wczytywanie danych z katalogu źródłowego: {dataset_root}...")
        print("Wnioskowanie...")
        print(f"Parametry: PRAWDOPODOBNA_MAX_ODLEGLOSC: {params['prawdopodobna_max_odlegloosc']}, PROG_PODOBIENSTWA_HIST: {params['prog_podobienstwa_hist']}")

    # Główna pętla wnioskowania
    # Pomija przetwarzanie pierwszej klatki, pustych klatek lub klatek następujących po pustych klatkach.
    for i in range(len(frames)):
        if i == 0:
            print(('-1 ' * frames[i][1]).strip())
            continue
        if frames[i][1] == 0:
            print()
            continue
        if frames[i - 1][1] == 0:
            print(('-1 ' * frames[i][1]).strip())
            continue

        frame1_bboxes = frames[i - 1][2]
        frame2_bboxes = frames[i][2]
        frame1_image = cv2.imread(os.path.join(dataset_root, 'frames', frames[i - 1][0]))
        frame2_image = cv2.imread(os.path.join(dataset_root, 'frames', frames[i][0]))

        frame1_histograms = [calculate_histogram(frame1_image[int(coord[1]):int(coord[1] + coord[3]),
                                                               int(coord[0]):int(coord[0] + coord[2])])
                             for coord in frame1_bboxes]
        frame2_histograms = [calculate_histogram(frame2_image[int(coord[1]):int(coord[1] + coord[3]),
                                                               int(coord[0]):int(coord[0] + coord[2])])
                             for coord in frame2_bboxes]

        G = build_factor_graph(frame1_bboxes, frame2_bboxes, frame1_histograms, frame2_histograms, params)
        matching = perform_inference(G)

        if with_ground_truth:
            correct, incorrect = accuracy_metric(list(matching.values()), [int(item) for item in ground_truths[i]])
            correct_cumulative += correct
            incorrect_cumulative += incorrect

        matching_output_view = list(matching.values())
        matching_output_view = ' '.join([str(item) for item in matching_output_view])
        print(f"{matching_output_view}")

        if verbose:
            print(f"matching: {matching}, ground_truth: {ground_truths[i]}")
            print(f"Single sample accuracy: {correct} / {correct + incorrect} = {correct / (correct + incorrect)}")

    if with_ground_truth and verbose:
        print(f"Total accuracy: {correct_cumulative} / {correct_cumulative + incorrect_cumulative} = {correct_cumulative / (correct_cumulative + incorrect_cumulative)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skrypt wnioskowania dla dopasowania indeksu bbox pomiędzy ramkami")
    parser.add_argument('dataset_root', type=str, help="Ścieżka do katalogu głównego zestawu danych")
    parser.add_argument('--with_ground_truth', action='store_true', help="Czy używać pliku bboxes_gt.txt do obliczenia dokładności")
    parser.add_argument('--verbose', action='store_true', help="Określa, czy drukować szczegółowe dane wyjściowe")

    args = parser.parse_args()
    main(args.dataset_root, args.with_ground_truth, args.verbose)
