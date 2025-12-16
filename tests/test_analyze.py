from review_summary.analyze import analyze_reviews, generate_summary


def test_analyze_simple():
    reviews = [
        "Great battery life and fast charging",
        "Battery lasts all day, excellent battery",
        "Camera quality is outstanding",
        "Camera struggles in low light",
        "Overheats while gaming sometimes",
        "Phone gets very hot when charging",
    ]

    result = analyze_reviews(reviews, top_n=3)
    # Expect battery and camera to appear in pros/cons depending on sentiment grouping
    pros = [p for p, _ in result.get("pros", [])]
    cons = [c for c, _ in result.get("cons", [])]

    # battery should appear among pros
    assert any("battery" in p for p in pros), f"battery not in pros: {pros}"
    # camera should appear in pros and/or cons depending on sentiment split
    assert any("camera" in p for p in pros + cons), f"camera not found in pros or cons: {pros} {cons}"
