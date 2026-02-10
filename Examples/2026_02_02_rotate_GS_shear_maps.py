"""
Notebook to rotate shear maps and verify that the rotation is done correctly.
Author: Sacha Guerrini
"""
# %%
import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')

import os
import time

# Trick to plot with tex
os.environ["LD_LIBRARY_PATH"] = ""
os.environ["CONDA_PREFIX"] = "/home/guerrini/.conda/envs/sp_validation_3.11"

import numpy as np
from astropy.io import fits
import healpy as hp

import matplotlib.pyplot as plt
import seaborn as sns


if os.path.exists("/home/guerrini/matplotlib_config/paper.mplstyle"):
    plt.style.use(
    "/home/guerrini/matplotlib_config/paper.mplstyle"
    )
# Set default palette - will be updated per plot as needed
sns.set_palette("deep")

if ipython is not None:
    ipython.run_line_magic('matplotlib', 'inline')

# %%
# Specify path to the data
path_shear_map = "/home/guerrini/euclid/LSS_forward_model/Data/rgamma12_true_tomobin_[1]_GS.npy.npy"
path_noise = "/home/guerrini/euclid/LSS_forward_model/Data/rgamma12_noisy_tomobin_[1]_RR2.npy"

# %%
# Load the shear map
shear_map = np.load(path_shear_map)

# Visualise the shear map
hp.mollview(shear_map.real, title="Shear map - real part", cmap="magma")
plt.show()

# %%
nside = hp.get_nside(shear_map)
print(f"Nside of the shear map: {nside}")
lmax = 2*nside

# Let's first check that we can go from maps to alm and back without loss
start = time.time()
almE, almB = hp.map2alm_spin(
    [shear_map.real, shear_map.imag],
    spin=2,
    lmax=lmax,
)

reconstructed_map_real, reconstructed_map_imag = hp.alm2map_spin(
    [almE, almB],
    nside=nside,
    spin=2,
    lmax=lmax
)
end = time.time()

print(f"Time taken for map2alm and alm2map: {end - start:.2f} seconds")

# %%
# Plot the histograms and check how it overlaps
plt.figure(figsize=(8, 6))

plt.hist(
    shear_map.real,
    bins=100,
    density=True,
    alpha=0.5,
    label="Original map - real part",
    histtype="step",
)
plt.hist(
    reconstructed_map_real,
    bins=100,
    density=True,
    alpha=0.5,
    label="Reconstructed map - real part",
    histtype="step",
)

plt.xlabel(r"Shear value $\gamma_1$")
plt.ylabel("Density")
plt.legend()
plt.show()

plt.figure()

plt.hist(
    shear_map.imag,
    bins=100,
    density=True,
    alpha=0.5,
    label="Original map - imag part",
    histtype="step",
)
plt.hist(
    reconstructed_map_imag,
    bins=100,
    density=True,
    alpha=0.5,
    label="Reconstructed map - imag part",
    histtype="step",
)

plt.xlabel(r"Shear value $\gamma_2$")
plt.ylabel("Density")
plt.legend()
plt.show()

# %%
# Visualise the map of the difference
delta_map_real = shear_map.real - reconstructed_map_real
delta_map_imag = shear_map.imag - reconstructed_map_imag

hp.mollview(delta_map_real, title="Difference map - real part", cmap="magma")
plt.show()

hp.mollview(delta_map_imag, title="Difference map - imag part", cmap="magma")
plt.show()

# The difference is quite significant given the amplitude of the shear
# %%
# Let's check if the power spectrum is similar
cl_original = hp.anafast(shear_map, lmax=lmax)
cl_reconstructed = hp.anafast(reconstructed_map_real + 1j*reconstructed_map_imag, lmax=lmax)

# %%
plt.figure(figsize=(8, 6))

plt.plot(
    cl_original,
    label="Original map - EE",
    color="C0"
)
plt.plot(
    cl_reconstructed,
    label="Reconstructed map - EE",
    color="C1",
    alpha=0.5
)

plt.legend()
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$C_\ell$")
plt.yscale('log')
plt.show()

# %%
# Let's check the impact of the rotation on the shear map
# We start with a rotation of 0 degrees
def get_rot(rot_angle_ra, rot_angle_dec):
    return hp.Rotator(deg=True, rot=[rot_angle_ra, rot_angle_dec])

null_rot = get_rot(0, 0)

# %%
# Rotate the alms
start = time.time()
almE_rot, almB_rot = null_rot.rotate_alm(almE, lmax=lmax), null_rot.rotate_alm(almB, lmax=lmax)
end = time.time()

print(f"Time taken for alm rotation: {end - start:.2f} seconds")

# %%
# Check that the rotated alms are equal
print(
    f"All rotated alm E equals to input: {np.isclose(almE, almE_rot).all()}\n",
    f"All rotated alm B equals to input: {np.isclose(almB, almB_rot).all()}"
)

# %%
# Reconstruct the rotated map and compare to the reconstructed map (to isolate from the reconstruction error)
reconstructed_map_rot_real, reconstructed_map_rot_imag = hp.alm2map_spin(
    [almE_rot, almB_rot],
    nside=nside,
    spin=2,
    lmax=lmax
)

# %%
# Plot the histograms and check how it overlaps
plt.figure(figsize=(8, 6))

plt.hist(
    reconstructed_map_real,
    bins=100,
    density=True,
    alpha=0.5,
    label="Reconstructed map - real part",
    histtype="step",
)
plt.hist(
    reconstructed_map_rot_real,
    bins=100,
    density=True,
    alpha=0.5,
    label="Rotated reconstructed map - real part",
    histtype="step",
)

plt.xlabel(r"Shear value $\gamma_1$")
plt.ylabel("Density")
plt.legend()
plt.show()

plt.figure()
plt.hist(
    reconstructed_map_imag,
    bins=100,
    density=True,
    alpha=0.5,
    label="Reconstructed map - imag part",
    histtype="step",
)
plt.hist(
    reconstructed_map_rot_imag,
    bins=100,
    density=True,
    alpha=0.5,
    label="Rotated reconstructed map - imag part",
    histtype="step",
)

plt.xlabel(r"Shear value $\gamma_2$")
plt.ylabel("Density")
plt.legend()
plt.show()

# %%
# Visualise the map of the difference
delta_map_rot_real = reconstructed_map_real - reconstructed_map_rot_real
delta_map_rot_imag = reconstructed_map_imag - reconstructed_map_rot_imag
hp.mollview(delta_map_rot_real, title="Difference map after null rotation - real part", cmap="magma")
plt.show()

hp.mollview(delta_map_rot_imag, title="Difference map after null rotation - imag part", cmap="magma")
plt.show()

# Tiny numerical effects 1e-16 level

# %%
# Now let's take a non-trivial rotation and check that the recovered map has the correct power spectrum
# and ellipticity distribution
rot_angle_ra, rot_angle_dec = 30, 45
non_null_rot = get_rot(rot_angle_ra, rot_angle_dec)

# %%
# Rotate the alms
start = time.time()
almE_rot, almB_rot = non_null_rot.rotate_alm(almE, lmax=lmax), non_null_rot.rotate_alm(almB, lmax=lmax)
end = time.time()

print(f"Time taken for alm rotation: {end - start:.2f} seconds")

# Check that the alms are different
print(
    f"Any rotated alm E different from input: {not np.isclose(almE, almE_rot).all()}\n",
    f"Any rotated alm B different from input: {not np.isclose(almB, almB_rot).all()}"
)

# %%
# Reconstruct the rotated map
reconstructed_map_rot_real, reconstructed_map_rot_imag = hp.alm2map_spin(
    [almE_rot, almB_rot],
    nside=nside,
    spin=2,
    lmax=lmax
)

# %%
# Plot the histograms and check how it overlaps
plt.figure(figsize=(8, 6))

plt.hist(
    reconstructed_map_real,
    bins=100,
    density=True,
    alpha=0.5,
    label="Reconstructed map - real part",
    histtype="step",
)
plt.hist(
    reconstructed_map_rot_real,
    bins=100,
    density=True,
    alpha=0.5,
    label="Rotated reconstructed map - real part",
    histtype="step",
)

plt.xlabel(r"Shear value $\gamma_1$")
plt.ylabel("Density")
plt.legend()
plt.show()

plt.figure()
plt.hist(
    reconstructed_map_imag,
    bins=100,
    density=True,
    alpha=0.5,
    label="Reconstructed map - imag part",
    histtype="step",
)
plt.hist(
    reconstructed_map_rot_imag,
    bins=100,
    density=True,
    alpha=0.5,
    label="Rotated reconstructed map - imag part",
    histtype="step",
)
plt.xlabel(r"Shear value $\gamma_2$")
plt.ylabel("Density")
plt.legend()
plt.show()

# The histogram agree with some shifts which is expected

# %%
# Visualise the map of the difference
delta_map_real = reconstructed_map_real - reconstructed_map_rot_real
delta_map_imag = reconstructed_map_imag - reconstructed_map_rot_imag
hp.mollview(delta_map_real, title="Difference map after non-null rotation - real part", cmap="magma")
plt.show()
hp.mollview(delta_map_imag, title="Difference map after non-null rotation - imag part", cmap="magma")
plt.show()

# This time we see structure due to the rotation.
# %%
# We can check that the cls are well preserved
cl_rotated = hp.anafast(reconstructed_map_rot_real + 1j*reconstructed_map_rot_imag, lmax=lmax)
cl_reconstructed = hp.anafast(reconstructed_map_real + 1j*reconstructed_map_imag, lmax=lmax)

# %%
plt.figure()

plt.plot(
    cl_reconstructed,
    label="Reconstructed map - EE",
    color="C0"
)
plt.plot(
    cl_rotated,
    label="Rotated reconstructed map - EE",
    color="C1",
    alpha=0.5
)

plt.legend()
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$C_\ell$")
plt.yscale('log')
plt.show()
# No problem here

# %%
# Finally, let's check that if we apply one rotation and its inverse
# we recover the original map

inv_rot = hp.Rotator(
    deg=True, inv=True, rot=[rot_angle_ra, rot_angle_dec]
)

# Rotate back the alms
start = time.time()
almE_inv_rot, almB_inv_rot = inv_rot.rotate_alm(almE_rot, lmax=lmax), inv_rot.rotate_alm(almB_rot, lmax=lmax)
end = time.time()

print(f"Time taken for inverse alm rotation: {end - start:.2f} seconds")

# Check that the alms are similar to the input
print(
    f"All inverse rotated alm E equals to input: {np.isclose(almE, almE_inv_rot).all()}\n",
    f"All inverse rotated alm B equals to input: {np.isclose(almB, almB_inv_rot).all()}"
)

# %%
# Reconstruct the map from the rotated back alms
reconstructed_inv_map_real, reconstructed_inv_map_imag = hp.alm2map_spin(
    [almE_inv_rot, almB_inv_rot],
    nside=nside,
    spin=2,
    lmax=lmax
)


# %%
# Plot the histograms and check how it overlaps
plt.figure(figsize=(8, 6))

plt.hist(
    reconstructed_map_real,
    bins=100,
    density=True,
    alpha=0.5,
    label="Reconstructed map - real part",
    histtype="step",
)
plt.hist(
    reconstructed_inv_map_real,
    bins=100,
    density=True,
    alpha=0.5,
    label="Inverse rotated reconstructed map - real part",
    histtype="step",
)

plt.xlabel(r"Shear value $\gamma_1$")
plt.ylabel("Density")
plt.legend()
plt.show()

plt.figure()

plt.hist(
    reconstructed_map_imag,
    bins=100,
    density=True,
    alpha=0.5,
    label="Reconstructed map - imag part",
    histtype="step",
)
plt.hist(
    reconstructed_inv_map_imag,
    bins=100,
    density=True,
    alpha=0.5,
    label="Inverse rotated reconstructed map - imag part",
    histtype="step",
)
plt.xlabel(r"Shear value $\gamma_2$")
plt.ylabel("Density")
plt.legend()
plt.show()

# %%
# Visualise the map of the difference
delta_inv_map_real = reconstructed_map_real - reconstructed_inv_map_real
delta_inv_map_imag = reconstructed_map_imag - reconstructed_inv_map_imag
hp.mollview(delta_inv_map_real, title="Difference map after inverse rotation - real part", cmap="magma")
plt.show()
hp.mollview(delta_inv_map_imag, title="Difference map after inverse rotation - imag part", cmap="magma")
plt.show()


# %%
# Summary:

"""
Something very clear is that the rotation of the alm is working as expected and is not the problem.
The problem is the first step when going from map to alm. When we do this step, we solve an inverse problem and healpy might return a non-optimal solution. The only path forward to solve this method would to try and see if there is an alternative apprach to go from map to alm that would not rely on healpy basic routine and that would be more accurate.
"""