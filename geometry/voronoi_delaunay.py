import cv2
import numpy as np

WIDTH = 360
HEIGHT = 240


def main():
    points = []
    for i in range(30):
        v = np.random.rand(2) * np.array([WIDTH, HEIGHT])
        points.append(v.astype(np.uint16))

    subdiv = cv2.Subdiv2D()
    subdiv.initDelaunay((0, 0, WIDTH, HEIGHT))
    subdiv.insert(points)

    # generate voronoi facets
    (facetLists, centerPoints) = subdiv.getVoronoiFacetList(None)

    # visualize voronoi diagram
    img = 255 * np.ones((HEIGHT, WIDTH, 3), np.uint8)
    for (lines, center) in zip(facetLists, centerPoints):
        img = cv2.circle(img, tuple(center), 1, (0, 0, 255))
        for i in range(len(lines)):
            p1 = lines[i-1].round().astype(np.int16)
            p2 = lines[i].round().astype(np.int16)
            img = cv2.line(img, tuple(p1), tuple(p2), (0, 0, 0))

    cv2.imshow('voronoi', img)

    # generate delaunay diaglram
    img = 255 * np.ones((HEIGHT, WIDTH, 3), np.uint8)
    for edge in subdiv.getEdgeList():
        p1 = (edge[0], edge[1])
        p2 = (edge[2], edge[3])
        img = cv2.line(img, p1, p2, (0, 0, 0))

    cv2.imshow('delaunay', img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
