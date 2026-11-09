"""Check the lease passed by reserve before each batch."""

from reserved import sdk
from time import sleep


if __name__ == "__main__":
    with sdk.adopt_inherited_lease() as reservation:
        for batch in range(1, 4):
            reservation.check()
            print(f"lease {reservation.reservation_id} checked before batch {batch}")
            sleep(1)  # Run one batch here.
