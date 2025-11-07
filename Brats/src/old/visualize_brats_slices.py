import os
import glob
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def robust_minmax_normalize(volume: np.ndarray) -> np.ndarray:
	"""Robust [0,1] normalization using 1st-99th percentiles over brain tissue."""
	brain_mask = volume > 0
	if not np.any(brain_mask):
		return volume
	brain_voxels = volume[brain_mask]
	p1, p99 = np.percentile(brain_voxels, [1, 99])
	if p99 <= p1:
		p1 = float(np.min(brain_voxels))
		p99 = float(np.max(brain_voxels))
	vol = np.clip(volume, p1, p99)
	vol = (vol - p1) / (p99 - p1 + 1e-8)
	vol[~brain_mask] = 0.0
	return vol


def discover_subjects(data_dir: str) -> List[str]:
	"""Return sorted list of subject directories like BraTS-GLI-* inside data_dir."""
	return sorted(glob.glob(os.path.join(data_dir, "BraTS-GLI-*")))


def resolve_paths_for_subject(subject_dir: str, modality: str) -> Tuple[str, str]:
	"""Find paths for the requested modality and segmentation for a given subject.

	Supported modality keys (per dataset naming seen in this repo):
	- t1n -> *-t1n.nii.gz
	- t1c -> *-t1c.nii.gz
	- t2w -> *-t2w.nii.gz
	- t2f -> *-t2f.nii.gz (FLAIR)
	"""
	pattern_map: Dict[str, str] = {
		"t1n": "*-t1n.nii.gz",
		"t1c": "*-t1c.nii.gz",
		"t2w": "*-t2w.nii.gz",
		"t2f": "*-t2f.nii.gz",
	}
	if modality not in pattern_map:
		raise ValueError(f"Unsupported modality '{modality}'. Use one of: {list(pattern_map.keys())}")
	img_candidates = glob.glob(os.path.join(subject_dir, pattern_map[modality]))
	seg_candidates = glob.glob(os.path.join(subject_dir, "*-seg.nii.gz"))
	if not img_candidates or not seg_candidates:
		raise FileNotFoundError(f"Missing files in {subject_dir} for modality={modality}")
	return img_candidates[0], seg_candidates[0]


def choose_tumor_slices(segmentation: np.ndarray, num_slices: int = 6, plane: str = "axial") -> List[int]:
	"""Choose slice indices that contain tumor for the requested plane.

	- plane axial: iterate along z (axis=2)
	- plane coronal: iterate along y (axis=1)
	- plane sagittal: iterate along x (axis=0)
	Select indices where tumor present and spread evenly over that range.
	"""
	assert plane in {"axial", "coronal", "sagittal"}
	mask = segmentation > 0
	if plane == "axial":
		counts = mask.sum(axis=(0, 1))
	elif plane == "coronal":
		counts = mask.sum(axis=(0, 2))
	else:
		counts = mask.sum(axis=(1, 2))
	indices = np.where(counts > 0)[0]
	if indices.size == 0:
		# fallback: pick middle slices
		size = segmentation.shape[2 if plane == "axial" else (1 if plane == "coronal" else 0)]
		return [int(round(x)) for x in np.linspace(size * 0.35, size * 0.65, num_slices)]
	if indices.size <= num_slices:
		return indices.tolist()
	# spread evenly across tumor-occupied indices
	selected = np.linspace(indices.min(), indices.max(), num_slices)
	return [int(round(x)) for x in selected]


def get_slice(volume: np.ndarray, index: int, plane: str) -> np.ndarray:
	"""Extract a 2D slice from a 3D volume for the given plane and index."""
	if plane == "axial":
		return volume[:, :, index]
	if plane == "coronal":
		return volume[:, index, :]
	return volume[index, :, :]


def label_color_map(unique_labels: List[int]) -> Dict[int, Tuple[float, float, float]]:
	"""Return an RGB color map (0-1 floats) for segmentation labels (excluding 0)."""
	# BraTS labels: 1=NCR/NET (necrotic/non-enhancing core), 2=ED (edema), 4=ET (enhancing tumor). 3 optional.
	defaults = {
		1: (0.90, 0.10, 0.10),  # red - necrotic/non-enhancing core (NCR/NET)
		2: (1.00, 0.80, 0.20),  # yellow - edema (ED)
		3: (0.80, 0.20, 0.80),  # magenta - optional
		4: (0.22, 0.49, 0.72),  # blue - enhancing tumor (ET)
	}
	return {lbl: defaults.get(int(lbl), (0.10, 0.90, 0.10)) for lbl in unique_labels}


def overlay_masks_on_slice(ax, base_slice: np.ndarray, seg_slice: np.ndarray, cmap: Dict[int, Tuple[float, float, float]], alpha: float = 0.4, title: Optional[str] = None) -> None:
	"""Render a grayscale base slice with colored segmentation overlays into ax."""
	ax.imshow(base_slice, cmap="gray", vmin=0.0, vmax=1.0)
	labels = [l for l in np.unique(seg_slice) if l != 0]
	for lbl in labels:
		color = cmap.get(int(lbl), (0.1, 0.9, 0.1))
		mask = (seg_slice == lbl).astype(np.float32)
		if np.any(mask):
			colored = np.zeros((*mask.shape, 4), dtype=np.float32)
			colored[..., 0] = color[0]
			colored[..., 1] = color[1]
			colored[..., 2] = color[2]
			colored[..., 3] = mask * alpha
			ax.imshow(colored)
	ax.set_axis_off()
	if title:
		ax.set_title(title, fontsize=10)


def save_single_slice(image_vol: np.ndarray, seg_vol: np.ndarray, plane: str, index: int, out_path: str, subject_name: str, modality_name: str, dpi: int = 300) -> None:
	"""Speichert eine einzelne Overlay-Ansicht (ein Bild pro PNG)."""
	assert plane in {"axial", "coronal", "sagittal"}
	# Clamp Index
	if plane == "axial":
		index = int(np.clip(index, 0, image_vol.shape[2] - 1))
	elif plane == "coronal":
		index = int(np.clip(index, 0, image_vol.shape[1] - 1))
	else:
		index = int(np.clip(index, 0, image_vol.shape[0] - 1))

	img_slice = get_slice(image_vol, index, plane)
	seg_slice = get_slice(seg_vol, index, plane)
	labels_present = [int(l) for l in np.unique(seg_slice) if l != 0]
	cmap = label_color_map(sorted(labels_present))

	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	overlay_masks_on_slice(ax, img_slice, seg_slice, cmap, alpha=0.45, title=f"{plane} index={index}")
	fig.suptitle(f"{subject_name} — {modality_name.upper()} ({plane})", fontsize=12)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fig.savefig(out_path, dpi=dpi)
	plt.close(fig)


def create_montage(image_vol: np.ndarray, seg_vol: np.ndarray, slice_indices: List[int], plane: str, out_path: str, subject_name: str, modality_name: str) -> None:
	"""Create and save a montage PNG for requested slices with mask overlays."""
	# Build a color map based on labels present in the selected slices
	labels_present = set()
	for idx in slice_indices:
		seg_slice = get_slice(seg_vol, idx, plane)
		labels_present.update([int(l) for l in np.unique(seg_slice) if l != 0])
	cmap = label_color_map(sorted(labels_present))

	cols = min(6, max(1, len(slice_indices)))
	rows = int(np.ceil(len(slice_indices) / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.3 * rows))
	if rows == 1 and cols == 1:
		axes = np.array([[axes]])
	elif rows == 1:
		axes = np.array([axes])
	elif cols == 1:
		axes = axes.reshape(rows, 1)

	for i, idx in enumerate(slice_indices):
		r = i // cols
		c = i % cols
		ax = axes[r, c]
		img_slice = get_slice(image_vol, idx, plane)
		seg_slice = get_slice(seg_vol, idx, plane)
		overlay_masks_on_slice(
			ax,
			img_slice,
			seg_slice,
			cmap,
			alpha=0.45,
			title=f"{plane} slice {idx}",
		)

	# Hide any unused axes
	for j in range(len(slice_indices), rows * cols):
		r = j // cols
		c = j % cols
		axes[r, c].set_visible(False)

	fig.suptitle(f"{subject_name} — {modality_name.upper()} with segmentation overlays ({plane})", fontsize=12)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def pick_subject_with_tumor(subject_dirs: List[str]) -> Optional[str]:
	"""Pick the first subject that has any tumor voxels (>0) in segmentation."""
	for s in subject_dirs:
		seg_candidates = glob.glob(os.path.join(s, "*-seg.nii.gz"))
		if not seg_candidates:
			continue
		try:
			seg = nib.load(seg_candidates[0]).get_fdata()
			if np.any(seg > 0):
				return s
		except Exception:
			continue
	return None


def find_top_slices_by_mask(segmentation: np.ndarray, plane: str, top_k: int = 3) -> List[int]:
	"""Gibt die Slice-Indizes mit den meisten Tumor-Pixeln (absteigend) zurück."""
	assert plane in {"axial", "coronal", "sagittal"}
	mask = segmentation > 0
	if plane == "axial":
		counts = mask.sum(axis=(0, 1))
	elif plane == "coronal":
		counts = mask.sum(axis=(0, 2))
	else:
		counts = mask.sum(axis=(1, 2))
	indices = np.argsort(counts)[::-1]  # absteigend
	indices = [int(i) for i in indices if counts[int(i)] > 0]
	return indices[:top_k]


def tumor_centroid_in_axial_slice(segmentation: np.ndarray, axial_index: int) -> Optional[Tuple[int, int]]:
	"""Berechne den (x,y)-Schwerpunkt der Tumormaske in einem axialen Slice."""
	if axial_index < 0 or axial_index >= segmentation.shape[2]:
		return None
	slice_mask = segmentation[:, :, axial_index] > 0
	coords = np.argwhere(slice_mask)
	if coords.size == 0:
		return None
	# coords sind (y, x); wir geben (x, y) zurück
	y_mean = int(round(coords[:, 0].mean()))
	x_mean = int(round(coords[:, 1].mean()))
	return x_mean, y_mean


def create_orthogonal_triptych(image_vol: np.ndarray, seg_vol: np.ndarray, axial_index: int, out_path: str, subject_name: str, modality_name: str) -> None:
	"""Erzeuge ein 1x3 Bild: coronal | axial | sagittal, zentriert auf Tumor im axialen Slice."""
	centroid = tumor_centroid_in_axial_slice(seg_vol, axial_index)
	if centroid is None:
		# Fallback: benutze Slice-Mitte
		x_c = image_vol.shape[0] // 2
		y_c = image_vol.shape[1] // 2
	else:
		x_c, y_c = centroid

	# Indizes für orthogonale Slices
	cor_index = int(np.clip(y_c, 0, image_vol.shape[1] - 1))  # coronal: y-Ebene
	sag_index = int(np.clip(x_c, 0, image_vol.shape[0] - 1))  # sagittal: x-Ebene

	# Gemeinsame Farbkarte über alle drei Slices
	labels_present = set()
	labels_present.update([int(l) for l in np.unique(seg_vol[:, :, axial_index]) if l != 0])
	labels_present.update([int(l) for l in np.unique(seg_vol[:, cor_index, :]) if l != 0])
	labels_present.update([int(l) for l in np.unique(seg_vol[sag_index, :, :]) if l != 0])
	cmap = label_color_map(sorted(labels_present))

	fig, axes = plt.subplots(1, 3, figsize=(12, 4))

	# Coronal (links)
	img_cor = image_vol[:, cor_index, :]
	seg_cor = seg_vol[:, cor_index, :]
	overlay_masks_on_slice(axes[0], img_cor, seg_cor, cmap, alpha=0.45, title=f"coronal y={cor_index}")

	# Axial (mitte)
	img_ax = image_vol[:, :, axial_index]
	seg_ax = seg_vol[:, :, axial_index]
	overlay_masks_on_slice(axes[1], img_ax, seg_ax, cmap, alpha=0.45, title=f"axial z={axial_index}")

	# Sagittal (rechts)
	img_sag = image_vol[sag_index, :, :]
	seg_sag = seg_vol[sag_index, :, :]
	overlay_masks_on_slice(axes[2], img_sag, seg_sag, cmap, alpha=0.45, title=f"sagittal x={sag_index}")

	fig.suptitle(f"{subject_name} — {modality_name.upper()} (tumor-zentriert)", fontsize=12)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def create_explicit_triptych(image_vol: np.ndarray, seg_vol: np.ndarray, x_index: int, y_index: int, z_index: int, out_path: str, subject_name: str, modality_name: str) -> None:
	"""Erzeuge ein 1x3 Bild mit EXAKTEN Indizes: coronal (y), axial (z), sagittal (x)."""
	# Clamp Indizes
	x_index = int(np.clip(x_index, 0, image_vol.shape[0] - 1))
	y_index = int(np.clip(y_index, 0, image_vol.shape[1] - 1))
	z_index = int(np.clip(z_index, 0, image_vol.shape[2] - 1))

	labels_present = set()
	labels_present.update([int(l) for l in np.unique(seg_vol[:, y_index, :]) if l != 0])
	labels_present.update([int(l) for l in np.unique(seg_vol[:, :, z_index]) if l != 0])
	labels_present.update([int(l) for l in np.unique(seg_vol[x_index, :, :]) if l != 0])
	cmap = label_color_map(sorted(labels_present))

	fig, axes = plt.subplots(1, 3, figsize=(12, 4))

	# Coronal (links) bei y_index
	img_cor = image_vol[:, y_index, :]
	seg_cor = seg_vol[:, y_index, :]
	overlay_masks_on_slice(axes[0], img_cor, seg_cor, cmap, alpha=0.45, title=f"coronal y={y_index}")

	# Axial (mitte) bei z_index
	img_ax = image_vol[:, :, z_index]
	seg_ax = seg_vol[:, :, z_index]
	overlay_masks_on_slice(axes[1], img_ax, seg_ax, cmap, alpha=0.45, title=f"axial z={z_index}")

	# Sagittal (rechts) bei x_index
	img_sag = image_vol[x_index, :, :]
	seg_sag = seg_vol[x_index, :, :]
	overlay_masks_on_slice(axes[2], img_sag, seg_sag, cmap, alpha=0.45, title=f"sagittal x={x_index}")

	fig.suptitle(f"{subject_name} — {modality_name.upper()} (explizite Slices)", fontsize=12)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fig.savefig(out_path, dpi=150)
	plt.close(fig)

def write_global_legend(output_root: str) -> None:
	legend_lines = [
		"Label-Farben (BraTS):",
		"1 -> Rot (Nekrotisch/Nicht-Enhancing Tumorkern, NCR/NET)",
		"2 -> Gelb (Peritumorales Ödem, ED)",
		"3 -> Magenta (optional, falls vorhanden)",
		"4 -> Blau (Enhancing Tumor, ET)",
		"",
		"Abgeleitete Regionen:",
		"Enhancing core (ET): Blau",
		"Peritumorales Ödem (ED): Gelb",
		"Totaler Tumor (WT = 1+2+4): Rot + Gelb + Blau",
	]
	os.makedirs(output_root, exist_ok=True)
	with open(os.path.join(output_root, "legend_colors.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(legend_lines) + "\n")


def main():
	parser = argparse.ArgumentParser(description="Erstellt pro Subject 5 Bilder mit farblichen Segmentierungs-Overlays (verschiedene Ebenen)")
	parser.add_argument("--data-dir", type=str, default=os.path.join("./datasets", "BraTS2025-GLI-PRE-Challenge-TrainingData"), help="Pfad zu BraTS2025 Trainingsdaten")
	parser.add_argument("--modality", type=str, default="t1c", choices=["t1n", "t1c", "t2w", "t2f"], help="Modalität für Hintergrund")
	parser.add_argument("--num-subjects", type=int, default=10, help="Anzahl Subjects (mit Tumor)")
	parser.add_argument("--output-root", type=str, default=os.path.join("visualizations"), help="Wurzelordner für Ausgaben")
	parser.add_argument("--dpi", type=int, default=300, help="DPI für PNG-Ausgaben (höher = schärfer)")
	# Fokus-Optionen für gezielte Ansicht (z.B. axial z=84)
	parser.add_argument("--focus-subject", type=str, default=None, help="Optional: Subjektname für gezielte orthogonale Ansicht")
	parser.add_argument("--focus-axial", type=int, default=None, help="Optional: axialer Slice-Index (z) für gezielte Ansicht")
	parser.add_argument("--focus-coronal", type=int, default=None, help="Optional: coronal Slice-Index (y) für explizite Ansicht")
	parser.add_argument("--focus-sagittal", type=int, default=None, help="Optional: sagittal Slice-Index (x) für explizite Ansicht")
	parser.add_argument("--only-focus", action="store_true", help="Nur Fokus-Ansicht(en) erzeugen, kein Batch-Output")
	args = parser.parse_args()

	data_dir = os.path.abspath(args.data_dir)
	if not os.path.isdir(data_dir):
		raise FileNotFoundError(f"Datenordner nicht gefunden: {data_dir}")

	subject_dirs = discover_subjects(data_dir)
	if not subject_dirs:
		raise RuntimeError(f"Keine Subjekte im Datenordner gefunden: {data_dir}")

	# Filtere Subjects mit Tumor, nimm die ersten N
	subjects_with_tumor: List[str] = []
	for s in subject_dirs:
		try:
			seg = nib.load(glob.glob(os.path.join(s, "*-seg.nii.gz"))[0]).get_fdata()
			if np.any(seg > 0):
				subjects_with_tumor.append(s)
			if len(subjects_with_tumor) >= args.num_subjects:
				break
		except Exception:
			continue
	if not subjects_with_tumor:
		raise RuntimeError("Keine Subjects mit Tumor gefunden.")

	output_root = os.path.abspath(args.output_root)
	write_global_legend(output_root)

	# Wenn expliziter Fokus vollständig angegeben ist und nur Fokus gewünscht ist: nur diese drei PNGs speichern
	if args.focus_subject and args.focus_axial is not None and args.focus_coronal is not None and args.focus_sagittal is not None and args.only_focus:
		matches = [s for s in subject_dirs if os.path.basename(s) == args.focus_subject]
		if not matches:
			raise RuntimeError(f"Subject {args.focus_subject} nicht gefunden in {data_dir}")
		subject_dir = matches[0]
		img_path, seg_path = resolve_paths_for_subject(subject_dir, args.modality)
		image_vol = nib.load(img_path).get_fdata()
		seg_vol = nib.load(seg_path).get_fdata()
		image_vol = robust_minmax_normalize(image_vol)
		subj_out = os.path.join(output_root, args.focus_subject)
		os.makedirs(subj_out, exist_ok=True)

		# Drei einzelne PNGs (jeweils 1 Bild pro Datei)
		save_single_slice(image_vol, seg_vol, plane="axial", index=int(args.focus_axial), out_path=os.path.join(subj_out, f"{args.focus_subject}_axial_z{int(args.focus_axial)}.png"), subject_name=args.focus_subject, modality_name=args.modality, dpi=args.dpi)
		save_single_slice(image_vol, seg_vol, plane="coronal", index=int(args.focus_coronal), out_path=os.path.join(subj_out, f"{args.focus_subject}_coronal_y{int(args.focus_coronal)}.png"), subject_name=args.focus_subject, modality_name=args.modality, dpi=args.dpi)
		save_single_slice(image_vol, seg_vol, plane="sagittal", index=int(args.focus_sagittal), out_path=os.path.join(subj_out, f"{args.focus_subject}_sagittal_x{int(args.focus_sagittal)}.png"), subject_name=args.focus_subject, modality_name=args.modality, dpi=args.dpi)

		print("✓ Fokus-Outputs gespeichert unter:", subj_out)
		return

	# Für jedes Subject: 5 Bilder mit verschiedenen Ebenen/Slices erstellen
	for subject_dir in subjects_with_tumor:
		subject_name = os.path.basename(subject_dir)
		img_path, seg_path = resolve_paths_for_subject(subject_dir, args.modality)
		image_vol = nib.load(img_path).get_fdata()
		seg_vol = nib.load(seg_path).get_fdata()
		image_vol = robust_minmax_normalize(image_vol)

		# Slice-Auswahl: 2 axial, 2 coronal, 1 sagittal
		axial_indices = choose_tumor_slices(seg_vol, num_slices=2, plane="axial")
		cor_indices = choose_tumor_slices(seg_vol, num_slices=2, plane="coronal")
		sag_indices = choose_tumor_slices(seg_vol, num_slices=1, plane="sagittal")

		# Ausgabeordner pro Subject
		subj_out = os.path.join(output_root, subject_name)
		os.makedirs(subj_out, exist_ok=True)

		# Speichere jeweils einzelne PNGs (5 Bilder)
		# 1) Axial 2 Slices in einer Montage
		create_montage(
			image_vol=image_vol,
			seg_vol=seg_vol,
			slice_indices=axial_indices,
			plane="axial",
			out_path=os.path.join(subj_out, f"{subject_name}_axial.png"),
			subject_name=subject_name,
			modality_name=args.modality,
		)

		# 2) Coronal 2 Slices
		create_montage(
			image_vol=image_vol,
			seg_vol=seg_vol,
			slice_indices=cor_indices,
			plane="coronal",
			out_path=os.path.join(subj_out, f"{subject_name}_coronal.png"),
			subject_name=subject_name,
			modality_name=args.modality,
		)

		# 3) Sagittal 1 Slice
		create_montage(
			image_vol=image_vol,
			seg_vol=seg_vol,
			slice_indices=sag_indices,
			plane="sagittal",
			out_path=os.path.join(subj_out, f"{subject_name}_sagittal.png"),
			subject_name=subject_name,
			modality_name=args.modality,
		)

		# 4-5) Zusätzlich zwei gemischte Ansichten (axial+coronal und axial+sagittal)
		mix1 = (axial_indices[:1] or sag_indices) + (cor_indices[:1] or sag_indices)
		create_montage(
			image_vol=image_vol,
			seg_vol=seg_vol,
			slice_indices=mix1,
			plane="axial",  # Titel-Bezug ist plane; Bilder zeigen jeweiligen Slice im plane-Kontext
			out_path=os.path.join(subj_out, f"{subject_name}_mixed_axial_coronal.png"),
			subject_name=subject_name,
			modality_name=args.modality,
		)

		mix2 = (axial_indices[:1] or cor_indices) + (sag_indices[:1] or cor_indices)
		create_montage(
			image_vol=image_vol,
			seg_vol=seg_vol,
			slice_indices=mix2,
			plane="axial",
			out_path=os.path.join(subj_out, f"{subject_name}_mixed_axial_sagittal.png"),
			subject_name=subject_name,
			modality_name=args.modality,
		)

		# Bonus: Orthogonale Triptychen für die Top-Tumor-Slices (max. 2)
		top_axial = find_top_slices_by_mask(seg_vol, plane="axial", top_k=2)
		for idx in top_axial:
			create_orthogonal_triptych(
				image_vol=image_vol,
				seg_vol=seg_vol,
				axial_index=idx,
				out_path=os.path.join(subj_out, f"{subject_name}_orthogonal_axial{idx}.png"),
				subject_name=subject_name,
				modality_name=args.modality,
			)

	# Optional: gezielte Fokus-Ansicht
	if args.focus_subject and (args.focus_axial is not None or (args.focus_coronal is not None and args.focus_sagittal is not None)):
		matches = [s for s in subject_dirs if os.path.basename(s) == args.focus_subject]
		if matches:
			subject_dir = matches[0]
			img_path, seg_path = resolve_paths_for_subject(subject_dir, args.modality)
			image_vol = nib.load(img_path).get_fdata()
			seg_vol = nib.load(seg_path).get_fdata()
			image_vol = robust_minmax_normalize(image_vol)
			subj_out = os.path.join(output_root, args.focus_subject)
			os.makedirs(subj_out, exist_ok=True)
			# Wenn alle drei expliziten Indizes vorhanden sind, nutze die exakte Triptychon-Ansicht
			if args.focus_axial is not None and args.focus_coronal is not None and args.focus_sagittal is not None:
				create_explicit_triptych(
					image_vol=image_vol,
					seg_vol=seg_vol,
					x_index=int(args.focus_sagittal),
					y_index=int(args.focus_coronal),
					z_index=int(args.focus_axial),
					out_path=os.path.join(subj_out, f"{args.focus_subject}_FOCUS_x{int(args.focus_sagittal)}_y{int(args.focus_coronal)}_z{int(args.focus_axial)}.png"),
					subject_name=args.focus_subject,
					modality_name=args.modality,
				)
			# Sonst falls nur axial gegeben: tumor-zentriertes Triptychon
			elif args.focus_axial is not None:
				create_orthogonal_triptych(
					image_vol=image_vol,
					seg_vol=seg_vol,
					axial_index=int(args.focus_axial),
					out_path=os.path.join(subj_out, f"{args.focus_subject}_FOCUS_axial{int(args.focus_axial)}.png"),
					subject_name=args.focus_subject,
					modality_name=args.modality,
				)

	print("✓ Fertig. Ausgaben unter:", output_root)


if __name__ == "__main__":
	main()


