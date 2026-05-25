# API

## GET /

Zwraca stronę główną z formularzem do wgrania pliku i listą ostatnich 10 analiz.

## POST /analyze

Przyjmuje plik JSON z danymi SpO2 i zwraca wynik analizy.

Żądanie - plik JSON wysłany jako formularz (multipart) lub jako surowe ciało zapytania.

Przykład wywołania przez curl:

    curl -X POST http://localhost:5000/analyze -F "file=@data/input.json"

Struktura pliku wejściowego:

    record_duration_hours - czas nagrania w godzinach (liczba dodatnia)
    spo2 - lista próbek, każda ma timestamp i value
    value - wartość SpO2 w procentach (0-100)
    timestamp - znacznik czasu, musi rosnąć monotonicznie

Przykład:

    {
      "record_duration_hours": 8.0,
      "spo2": [
        {"timestamp": 0.0, "value": 98.0},
        {"timestamp": 1.0, "value": 97.5}
      ]
    }

Odpowiedź 200:

    {
      "apnea_events": 3,
      "ahi": 3.0,
      "severity": "lagodna",
      "score": 95
    }

Znaczenie severity:
- brak - AHI poniżej 5
- lagodna - AHI 5 do 14
- umiarkowana - AHI 15 do 29
- ciezka - AHI 30 i więcej

Błędy:
- 400 - niepoprawny JSON lub dane nie spełniają schematu
- 408 - analiza trwała dłużej niż 30 sekund
- 500 - błąd serwera

## GET /history

Zwraca 10 ostatnich analiz jako tablicę JSON, od najnowszej.

Przykład wywołania:

    curl http://localhost:5000/history
