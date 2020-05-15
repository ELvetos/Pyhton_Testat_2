# -*- coding: utf-8 -*-
"""
Created on Fr May 15 08:08:57 2020

@author: Simeï Steiner
@filename: Testat 2

"""
from scipy.constants import epsilon_0
import numpy as np


def e_field(q_values, q_locations, points):
    """
    Computes the E-field on the given points

    Parameters
    ----------
    q_values : int, float, list or 1D-np.array
    - contains the Values of the "Ladungen"
    q_locations : list, 2D list, 1D-np.array or 2D-np.array
    - contains the locations of the "Ladungen"
    points : list, 2D list, 1D-np.array or 2D-np.array
    - contains the locations of the points which should be computed

    Returns
    -------
    E : 1D-np.array or 2D-np.array
    - contains the E-field on the computed points

    """
    q_val = np.array(q_values)
    q_loc = np.array(q_locations)
    poin = np.array(points)
    assert q_loc.size % 3 == 0, "q_location is invalid"
    assert poin.size % 3 == 0, "points is invalid"
    assert q_val.size == q_loc.size/3, "q_values not equal q_loc"
    if poin.ndim == 1 and q_loc.ndim == 1:
        r = poin - q_loc
        rn = np.linalg.norm(r)
        E = (q_val/(4*np.pi*epsilon_0*rn**3))*r
    elif poin.ndim == 1 and q_loc.ndim > 1:
        E = np.array([0.0, 0.0, 0.0])
        for j in range(q_loc.shape[0]):
            r = poin-q_loc[j]
            # R = np.append(R, r.reshape(1,-1), axis = 0)
            rn = np.linalg.norm(r)
            E += (q_val[j]/(4*np.pi*epsilon_0*rn**3))*r
    else:
        E = np.empty((0, 3), float)
        for i in range(poin.shape[0]):
            e = np.array([0.0, 0.0, 0.0])
            for j in range(q_loc.shape[0]):
                r = poin[i]-q_loc[j]
                # R = np.append(R, r.reshape(1,-1), axis = 0)
                rn = np.linalg.norm(r)
                e += (q_val[j]/(4*np.pi*epsilon_0*rn**3))*r
            E = np.append(E, e.reshape(1, -1), axis=0)
    return E


def voltage(e_vectors, path):
    """
    computes the voltage along a given path in 3D-space

    Parameters
    ----------
    e_vectors : 2D-list or 2D-np.array
    -contains the E-field vectors
    path : 2D-list or 2D-np.array
    -contains the location of the E-field vectors

    Returns
    -------
    Un : float
    -contains the voltage Uab along the path.

    """
    e_vec = np.array(e_vectors)
    pa = np.array(path)
    assert e_vec.size == pa.size, "path and e_vector size are not equal"
    Un = 0
    for i in range(e_vec.shape[0]-1):
        de = (e_vec[i] + e_vec[i+1])/2
        Un += de.dot((pa[i+1]-pa[i]))
    return Un


def gauss_law(e_func, p1, p2, n_res=100):
    """

    Parameters
    ----------
    e_func : function
        Hiermit wird eine Referenz auf eine externe Funktion
        angegeben, welche die elektrischen Feldst¨arkevektoren E~
        an den gewunschten Punkten ¨ p im Raum zuruckgibt.
    p1 : list or 1D-np.array
        Hiermit werden die kartesischen Koordinaten des ersten
        Punktes P1 vom Quader angegeben.
    p2 : list or 1D-np.Array
        Hiermit werden die kartesischen Koordinaten des zweiten
        Punktes P2 vom Quader angegeben.
    n_res : int
        Unterteilung pro dimension. The default is 100.

    Returns
    -------
    fluss : float
        Hiermit wird der Gesamtfluss Φ durch die Oberfl¨ache S
        zuruckgegeben.

    """
    P1 = np.array(p1)
    P2 = np.array(p2)
    assert n_res > 0, "n_res is equal 0 or smaller"
    assert P1.size == 3, "p1 is not a 3D-point"
    assert P2.size == 3, "p2 is not a 3D-point"
    fluss = 0
    d = (P2 - P1) / n_res
    print(d)
    S1 = (d[0] * d[1]) * np.array([0, 0, 1])
    S2 = (d[0] * d[2]) * np.array([0, 1, 0])
    S3 = (d[1] * d[2]) * np.array([1, 0, 0])
    px = P1 + np.array([0, d[1] / 2, d[2] / 2])
    for i in range(n_res):
        for j in range(n_res):
            # print(px)
            fluss += e_func(px).dot(-S3)
            px += np.array([0, 0, d[2]])
        px -= np.array([0, 0, (n_res)*d[2]])
        px += np.array([0, d[1], 0])
    px = P1 + np.array([0, d[1] / 2, d[2] / 2])
    px[0] = P2[0]
    for i in range(n_res):
        for j in range(n_res):
            # print(px)
            fluss += e_func(px).dot(S3)
            px += np.array([0, 0, d[2]])
        px -= np.array([0, 0, (n_res)*d[2]])
        px += np.array([0, d[1], 0])
    py = P1 + np.array([d[0] / 2, 0, d[2] / 2])
    for i in range(n_res):
        for j in range(n_res):
            fluss += e_func(py).dot(-S2)
            # print(py)
            py += np.array([0, 0, d[2]])
        py -= np.array([0, 0, (n_res)*d[2]])
        py += np.array([d[0], 0, 0])
    py = P1 + np.array([d[0] / 2, 0, d[2] / 2])
    py[1] = P2[1]
    for i in range(n_res):
        for j in range(n_res):
            fluss += e_func(py).dot(S2)
            # print(py)
            py += np.array([0, 0, d[2]])
        py -= np.array([0, 0, (n_res)*d[2]])
        py += np.array([d[0], 0, 0])
    pz = P1 + np.array([d[0] / 2, d[1] / 2, 0])
    for i in range(n_res):
        for j in range(n_res):
            fluss += e_func(pz).dot(-S1)
            # print(pz)
            pz += np.array([d[0], 0, 0])
        pz -= np.array([(n_res)*d[0], 0, 0])
        pz += np.array([0, d[1], 0])
    pz = P1 + np.array([d[0] / 2, d[1] / 2, 0])
    pz[2] = P2[2]
    for i in range(n_res):
        for j in range(n_res):
            # print(pz)
            fluss += e_func(pz).dot(S1)
            pz += np.array([d[0], 0, 0])
        pz -= np.array([(n_res)*d[0], 0, 0])
        pz += np.array([0, d[1], 0])
    if np.sign(d[0]*d[1]*d[2]) == 1:
        return fluss
    else:
        return fluss * -1
