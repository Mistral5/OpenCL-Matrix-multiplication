To run the code specify:
1. Device to be used
2. Input file name
3. Output file name
4. Operating mode:
	- `1` Without local device memory
	- `2` With using local device memory
	- `3` With using local device memory and vectorization

Examples:
- 0 11x13x17.txt 11x13x17_out.txt 3
- 1 101x101x101.txt 01x101x101_out.txt 2
- 2 1Kx1Kx1K.txt 1Kx1Kx1K_out.txt 1