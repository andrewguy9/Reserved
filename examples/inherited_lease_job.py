"""Check the inherited lease before each batch, or run directly without one."""

import os

from reserved import sdk


def run_batches(reservation: sdk.Reservation | None = None) -> None:
    for batch in range(1, 4):
        if reservation is not None:
            reservation.check()
        print(f"processed batch {batch}")


def main() -> None:
    # The launcher sets RESERVED_ID even if the descriptor is missing.
    if "RESERVED_ID" in os.environ or "RESERVED_FD" in os.environ:
        with sdk.adopt_inherited_lease() as reservation:
            print(f"using reservation {reservation.reservation_id}")
            run_batches(reservation)
    else:
        print("running without a reservation")
        run_batches()


if __name__ == "__main__":
    main()
