// === Figure 위치 매핑 (order 기준) ===
export const FIGURE_MAP: Record<string, number> = {
  "easy_paragraph_1_4": 1,  // Fig.1 YOLO Detection System
  "easy_paragraph_2_1": 2,  // Fig.2 The Model
  "easy_paragraph_3_1": 3,  // Fig.3 The Architecture
  "easy_paragraph_7_1": 4,  // Fig.4 Error Analysis: Fast R-CNN vs. YOLO
  "easy_paragraph_9_1": 5,  // Fig.5 Picasso Dataset PR Curves
  "easy_paragraph_13_2": 6, // Fig.6 Qualitative Results
};

// === Figure 캡션 하드코딩 ===
export const FIGURE_CAPTION: Record<number, string> = {
  1: "Figure 1. The YOLO Detection System.",
  2: "Figure 2. The Model.",
  3: "Figure 3. The Architecture.",
  4: "Figure 4. Error Analysis: Fast R-CNN vs. YOLO.",
  5: "Figure 5. Picasso Dataset precision-recall curves.",
  6: "Figure 6. Qualitative Results.",
};