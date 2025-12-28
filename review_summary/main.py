"""CLI entrypoint for review summary."""
import sys
from .analyze import load_reviews, analyze_reviews, generate_summary


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: python -m review_summary.main <path-to-reviews.txt>")
        sys.exit(2)

    path = argv[0]
    reviews = load_reviews(path)
    
    # Validate that we have at least 50 reviews
    review_count = len(reviews)
    if review_count < 50:
        print(f"Warning: Expected at least 50 reviews, but found {review_count}.", file=sys.stderr)
    else:
        print(f"Successfully loaded {review_count} smartphone reviews.", file=sys.stderr)
    
    result = analyze_reviews(reviews, top_n=3)
    paragraph = generate_summary(result.get("pros", []), result.get("cons", []))
    print(paragraph)


if __name__ == "__main__":
    main()
