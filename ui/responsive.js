// Whether the annotator page is too small/narrow to label on (hawk-ai-internal: "people need to be doing the
// annotations in laptop not mobile"). A pure function of width, not device type: a tablet in landscape at a
// typical resolution (1024px+) already passes, same as a laptop; only phone-width (or a deliberately narrowed
// desktop window) is blocked. No DOM access, so it can be tested under plain Node (tests/test_responsive.py).
export const MIN_ANNOTATOR_WIDTH = 900;

export function tooSmallForAnnotating(web, width) {
  return Boolean(web) && width < MIN_ANNOTATOR_WIDTH;
}
