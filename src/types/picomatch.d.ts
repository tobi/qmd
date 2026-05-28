declare module "picomatch" {
  export type Matcher = (input: string) => boolean;
  export default function picomatch(pattern: string | string[], options?: Record<string, unknown>): Matcher;
}
