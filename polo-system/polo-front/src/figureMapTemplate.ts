// === Figure 위치 매핑 (figures_map.json order 기준) ===
export const FIGURE_MAP: Record<string, number> = {
  "easy_paragraph_2_3": 1,  // Fig.1 YOLO Detection System (system_p1.png) - order 1
  "easy_paragraph_3_2": 2,  // Fig.2 The Model (model_p1.png) - order 2  
  "easy_paragraph_3_7": 3,  // Fig.3 The Architecture (net_p1.png) - order 3
  "easy_paragraph_3_8": 4,  // Fig.4 Error Analysis (pie_compare_p1.png) - order 4
  "easy_paragraph_12_1": 5,  // Fig.5 Picasso Dataset (cubist_p1.png) - order 5
  "easy_paragraph_12_2": 6,  // Fig.6 Qualitative Results (art.jpg) - order 6
};

// === Figure 캡션 하드코딩 (figures_map.json caption 기준) ===
export const FIGURE_CAPTION: Record<number, string> = {
  1: "Figure 1. The YOLO Detection System.",
  2: "Figure 2. The Model.", 
  3: "Figure 3. The Architecture.",
  4: "Figure 4. Error Analysis: Fast R-CNN vs. YOLO.",
  5: "Figure 5. Picasso Dataset precision-recall curves.",
  6: "Figure 6. Qualitative Results.",
};