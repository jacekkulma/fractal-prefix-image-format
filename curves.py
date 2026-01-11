from abc import ABC, abstractmethod

class SpaceFillingCurve(ABC):
    """
    Abstract Base Class for all space filling curves.
    """
    
    @abstractmethod
    def d_to_xy(self, distance, order):
        """
        Maps a 1D distance to 2D coordinates (x, y).
        Order determines the grid size (2^order x 2^order).
        """
        pass

    @abstractmethod
    def xy_to_d(self, x, y, order):
        """
        Maps 2D coordinates (x, y) to a 1D distance.
        """
        pass
    
    def get_max_distance(self, order):
        """Returns the total number of points in the curve."""
        return (2**order) * (2**order)


class HilbertCurve(SpaceFillingCurve):
    """
    Implementation of the Hilbert Curve.
    Uses iterative approach to map distance to coordinates.
    """
    
    def d_to_xy(self, distance, order):
        x = 0
        y = 0
        s = 1
        
        while s < (1 << order):
            rx = 1 & (distance // 2)
            ry = 1 & (distance ^ rx)
            
            # Rotate/flip quadrant
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                # Swap x and y
                x, y = y, x
            
            x += s * rx
            y += s * ry
            distance //= 4
            s *= 2
            
        return x, y

    def xy_to_d(self, x, y, order):
        d = 0
        s = (1 << (order - 1))
        
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            
            d += s * s * ((3 * rx) ^ ry)
            
            if ry == 0:
                if rx == 1:
                    x = (1 << order) - 1 - x
                    y = (1 << order) - 1 - y
                x, y = y, x
                
            s //= 2
            
        return d


class ZOrderCurve(SpaceFillingCurve):
    """
    Implementation of the Z-Order (Morton) Curve.
    Simple bit-interleaving logic.
    """
    
    def _part1by1(self, n):
        n &= 0x0000FFFF
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    def _unpart1by1(self, n):
        n &= 0x55555555
        n = (n ^ (n >> 1)) & 0x33333333
        n = (n ^ (n >> 2)) & 0x0F0F0F0F
        n = (n ^ (n >> 4)) & 0x00FF00FF
        n = (n ^ (n >> 8)) & 0x0000FFFF
        return n

    def d_to_xy(self, distance, order):
        # De-interleave bits
        x = self._unpart1by1(distance)
        y = self._unpart1by1(distance >> 1)
        return x, y

    def xy_to_d(self, x, y, order):
        # Interleave bits
        return (self._part1by1(y) << 1) | self._part1by1(x)


class ScanlineCurve(SpaceFillingCurve):
    """
    Standard Raster Scan (Row by Row).
    Used as a baseline for comparison.
    """
    
    def d_to_xy(self, distance, order):
        width = 1 << order
        y = distance >> order  # equivalent to: distance // width
        x = distance & (width - 1)  # equivalent to: distance % width
        return x, y

    def xy_to_d(self, x, y, order):
        return (y << order) | x
