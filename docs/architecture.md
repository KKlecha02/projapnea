# Architektura

Użytkownik wysyła plik JSON przez formularz na stronie. Flask odbiera plik, sprawdza czy dane są poprawne i przekazuje je do algorytmu detekcji. Wynik jest zapisywany do pliku i wyświetlany na stronie z wykresem.

## Jak działa analiza

1. Aplikacja odbiera plik JSON z danymi SpO2
2. Sprawdza poprawność danych (schemat, znaczniki czasu)
3. Uruchamia algorytm detekcji z limitem czasu 30 sekund
4. Algorytm filtruje sygnał, szuka epizodów bezdechu i liczy AHI
5. Wynik trafia na stronę z wykresem

## Pliki i co robią

src/apnea.py - główny algorytm detekcji, same reguły, żadnych zależności ML

src/apnea_engine.py - model CNN-LSTM, używany tylko przy trenowaniu i walidacji

src/app.py - serwer Flask, trasy, walidacja wejścia, zapis historii

load_dataset.py - wczytuje dane pacjentów z plików CSV i JSON

train.py - trenuje model CNN-LSTM i zapisuje go do model.pt

validate.py - testuje algorytm na zbiorze walidacyjnym i liczy metryki

## Historia analiz

Każdy wynik jest zapisywany jako osobny plik JSON w folderze results. Endpoint /history zwraca 10 ostatnich analiz.
