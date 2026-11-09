"""Check the lease passed by reserve before each batch."""

from reserved import sdk


if __name__ == "__main__":
    with sdk.adopt_inherited_lease() as reservation:
        for batch in range(1, 4):
            reservation.check()
            print(f"lease {reservation.reservation_id} checked before batch {batch}")
            # Run one batch here.
