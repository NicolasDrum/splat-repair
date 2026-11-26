interface DeepFillSideInfo {
  offAxisGeo?: number | null;
  offAxisDeep?: number | null;
  deep_charts?: readonly unknown[];
}

export function formatSideError(info: DeepFillSideInfo): string {
  const { offAxisGeo, offAxisDeep, deep_charts } = info;
  if (offAxisGeo == null || offAxisDeep == null) return '';
  // The engine reports the first repaired face for a multi-chart fill.
  const face = (deep_charts?.length ?? 0) > 1 ? ' (first repaired face)' : '';
  // Each value is already averaged over pixels and color channels.
  const before = (100 * offAxisGeo).toFixed(1);
  const after = (100 * offAxisDeep).toFixed(1);
  return ` · from the side${face} ${before} → ${after}%`;
}
