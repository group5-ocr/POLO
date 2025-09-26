declare module 'react-katex' {
  import { ReactNode } from 'react';

  interface InlineMathProps {
    math: string;
    children?: ReactNode;
  }

  interface BlockMathProps {
    math: string;
    children?: ReactNode;
  }

  export const InlineMath: React.FC<InlineMathProps>;
  export const BlockMath: React.FC<BlockMathProps>;
}
