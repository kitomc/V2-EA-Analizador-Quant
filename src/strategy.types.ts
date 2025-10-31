
export interface RawStrategy {
  id: number;
  magicNumber: number;
  symbol: string;
  period: string;
  initialAccount: number;
  backtestStats: BacktestStats;
  equity: number[];
  balance: number[];
  originKey?: string;
}

export interface BacktestStats {
  profit: number;
  maxDrawdownPercent: number;
  countOfTrades: number;
  winLossRatio: number;
  profitFactor: number;
  maxConsecutiveLosses: number;
  rSquared: number;
  systemQualityNumber: number;
  sharpeRatio: number;
  returnToDrawdown: number;
  averagePositionLength: number;
}

export interface ProcessedStrategy extends RawStrategy {
  // Calculated Metrics
  calmarRatio: number;
  sortinoRatio: number;
  expectancy: number;
  winRate: number;
  robustnessScore: number;
  stabilityIndex: number;
  edgeQuality: number;
  profitMode: number;
  profitPercentile80: number;
  avgTradeDuration: number;
  treynorRatio?: number;
  beta?: number;
  zScore: number;
  iqr: number;
  // Synthetic scores for robustness
  monkeyScore: number;
  consistencyScore: number;
  mitigationScore?: number;
}

export interface Filters {
  minWinRate: number;
  minExpectancy: number;
  minProfitMode: number;
  minProfitPercentile80: number;
  maxAvgTradeDuration: number;
  minZScore: number;
  maxIqr: number;
}

export interface Portfolio {
  strategies: ProcessedStrategy[];
  equityCurve: number[];
  sharpeRatio: number;
  calmarRatio: number;
  treynorRatio: number;
  maxDrawdownPercent: number;
  avgProfitFactor: number;
  avgWinRate: number;
  totalTrades: number;
  profitMode: number;
  profitPercentile80: number;
  recoveryFactor: number;
  maxConsecutiveLosses: number;
}

export type SortKey = keyof BacktestStats | keyof ProcessedStrategy | 'id' | 'mitigationScore' | 'treynorRatio' | 'zScore' | 'iqr';
export type SortDirection = 'asc' | 'desc';

export interface PortfolioSegmentMetrics {
  sharpeRatio: number;
  calmarRatio: number;
  maxDrawdownPercent: number;
  returns: number;
}

export interface WalkForwardResult {
  status: 'robust' | 'fragile';
  inSample: PortfolioSegmentMetrics;
  outOfSample: PortfolioSegmentMetrics;
}