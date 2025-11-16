import { ChangeDetectionStrategy, Component, signal, computed, ElementRef, viewChild, effect } from '@angular/core';
import { CommonModule } from '@angular/common';

import { RawStrategy, ProcessedStrategy, Filters, Portfolio, SortKey, SortDirection, BacktestStats, WalkForwardResult, PortfolioSegmentMetrics } from './strategy.types';

declare var d3: any;

const USER_DEFAULTS_KEY = 'robustPortfolio_userDefaults';

// Type aliases for portfolio metrics and Monte Carlo results
type AnalyzablePortfolioMetric =
  | 'sharpeRatio'
  | 'calmarRatio'
  | 'treynorRatio'
  | 'maxDrawdownPercent'
  | 'avgProfitFactor'
  | 'profitMode'
  | 'profitPercentile80'
  | 'recoveryFactor'
  | 'maxConsecutiveLosses';

interface MonteCarloMetrics {
  sharpeRatio: number;
  calmarRatio: number;
  treynorRatio: number;
  maxDrawdownPercent: number;
  recoveryFactor: number;
  profitMode: number;
  profitPercentile80: number;
  maxConsecutiveLosses: number;
  rSquared: number;
}

interface MonteCarloResult {
  p5: MonteCarloMetrics;
  p50: MonteCarloMetrics;
  p95: MonteCarloMetrics;
}


@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {

  // Chart Elements
  portfolioChart = viewChild<ElementRef>('portfolioChart');
  strategyEquityChart = viewChild<ElementRef>('strategyEquityChart');
  strategyDrawdownChart = viewChild<ElementRef>('strategyDrawdownChart');
  correlationHeatmap = viewChild<ElementRef>('correlationHeatmap');
  ddReturnsScatter = viewChild<ElementRef>('ddReturnsScatter');
  winRateExpectancyScatter = viewChild<ElementRef>('winRateExpectancyScatter');
  robustnessMitigationScatter = viewChild<ElementRef>('robustnessMitigationScatter');

  // Raw and Processed Data Signals
  rawStrategies = signal<RawStrategy[]>([]);
  baseProcessedStrategies = computed<ProcessedStrategy[]>(() => {
    return this.rawStrategies().map(strat => this.processStrategy(strat));
  });

  processedStrategies = computed<ProcessedStrategy[]>(() => {
    const strats = this.baseProcessedStrategies();
    const selected = this.selectedStrategies();
    
    if (strats.length === 0) {
      return [];
    }
    
    // Use selected portfolio as benchmark if it has > 1 strategy, otherwise use all strategies
    const marketPortfolio = selected.length > 1 ? selected : strats;

    // Fallback if there's no meaningful market portfolio
    if (marketPortfolio.length < 2) {
      return strats.map(s => ({ ...s, treynorRatio: 0, beta: 1, mitigationScore: 50 }));
    }

    const marketReturnsLists = marketPortfolio.map(s => this.calculateReturns(s.equity));
    // FIX: Cast `r` to `number[]` because it is inferred as `unknown`.
    const minMarketReturnLength = Math.min(...marketReturnsLists.map(r => (r as number[]).length));

    // Fallback if market portfolio strategies have no returns history
    if (minMarketReturnLength < 2) {
        return strats.map(s => ({ ...s, treynorRatio: 0, beta: 1, mitigationScore: 50 }));
    }
    
    // Calculate the market's average returns
    const marketReturns: number[] = [];
    for (let i = 0; i < minMarketReturnLength; i++) {
      let sum = 0;
      for (const stratReturns of marketReturnsLists) {
        // This is safe because of the minMarketReturnLength check above
        sum += (stratReturns as number[])[i];
      }
      marketReturns.push(sum / marketReturnsLists.length);
    }
    
    // Pre-calculate all strategy returns to avoid re-calculation in loop
    const allStratReturnsMap = new Map(strats.map(s => [s.id, this.calculateReturns(s.equity)]));

    // Calculate metrics for each strategy against the market portfolio
    return strats.map((strat) => {
      const stratReturnsFull = allStratReturnsMap.get(strat.id) || [];
      
      // Align return series lengths
      const minLength = Math.min(marketReturns.length, (stratReturnsFull as number[]).length);
      
      if (minLength < 2) {
          return { ...strat, treynorRatio: 0, beta: 1, mitigationScore: 50 };
      }

      // FIX: Cast `stratReturnsFull` to `number[]` because it is inferred as `unknown`.
      const stratReturns = (stratReturnsFull as number[]).slice(0, minLength);
      const marketReturnsSliced = marketReturns.slice(0, minLength);
      
      const stratMean = this.mean(stratReturns);
      const marketMeanSliced = this.mean(marketReturnsSliced);
      const marketVarianceSliced = this.variance(marketReturnsSliced, marketMeanSliced);

      if (marketVarianceSliced === 0) {
        return { ...strat, treynorRatio: 0, beta: 1, mitigationScore: 50 };
      }

      // Calculate Beta and Treynor Ratio
      const covariance = this.covariance(stratReturns, marketReturnsSliced, stratMean, marketMeanSliced);
      const beta = covariance / marketVarianceSliced;
      const annualizedReturn = stratMean * 252;
      const treynorRatio = beta !== 0 ? annualizedReturn / beta : 0;

      // Calculate Correlation and Mitigation Score
      const stratStdev = this.stdev(stratReturns, stratMean);
      const marketStdev = this.stdev(marketReturnsSliced, marketMeanSliced);
      const correlation = this.correlation(stratReturns, marketReturnsSliced, stratMean, marketMeanSliced, stratStdev, marketStdev);
      const mitigationScore = (1 - (isNaN(correlation) ? 1 : correlation)) * 50;
      
      return { ...strat, treynorRatio, beta, mitigationScore };
    });
  });

  // Filter and Sorting Signals
  baseDefaultFilters: Filters = {
    minWinRate: 0.0,
    minExpectancy: 0,
    minProfitMode: 2,
    minProfitPercentile80: 4,
    maxAvgTradeDuration: 100,
    minZScore: 0,
    maxIqr: 100,
    minRSquared: 45,
    minTreynorRatio: 0,
  };
  effectiveDefaults: Filters;
  filters = signal<Filters>(this.baseDefaultFilters);
  sort = signal<{ key: SortKey; direction: SortDirection }>({ key: 'robustnessScore', direction: 'desc' });

  // Computed Signals for Display
  filteredStrategies = computed(() => {
    const f = this.filters();
    return this.processedStrategies()
      .filter(s => 
        s.winRate >= f.minWinRate &&
        s.expectancy >= f.minExpectancy &&
        s.profitMode >= f.minProfitMode &&
        s.profitPercentile80 >= f.minProfitPercentile80 &&
        s.avgTradeDuration <= f.maxAvgTradeDuration &&
        s.zScore >= f.minZScore &&
        s.iqr <= f.maxIqr &&
        s.backtestStats.rSquared >= f.minRSquared &&
        (s.treynorRatio ?? -Infinity) >= f.minTreynorRatio
      );
  });
  
  sortedStrategies = computed(() => {
    const s = this.sort();
    const strategies = [...this.filteredStrategies()];
    strategies.sort((a, b) => {
      let valA: any, valB: any;

      if (s.key in a) {
        valA = a[s.key as keyof ProcessedStrategy];
        valB = b[s.key as keyof ProcessedStrategy];
      } else if (s.key in a.backtestStats) {
        valA = a.backtestStats[s.key as keyof BacktestStats];
        valB = b.backtestStats[s.key as keyof BacktestStats];
      }

      if (valA === undefined || valA === null) valA = -Infinity;
      if (valB === undefined || valB === null) valB = -Infinity;
      
      if (valA < valB) return s.direction === 'asc' ? -1 : 1;
      if (valA > valB) return s.direction === 'asc' ? 1 : -1;
      return 0;
    });
    return strategies;
  });

  // Selection Signals
  selectedStrategiesMap = signal<Map<number, ProcessedStrategy>>(new Map());
  selectedStrategies = computed(() => Array.from(this.selectedStrategiesMap().values()));
  
  areAllFilteredSelected = computed(() => {
    const filtered = this.filteredStrategies();
    const selected = this.selectedStrategiesMap();
    if (filtered.length === 0) return false;
    return filtered.every(s => selected.has(s.id));
  });

  // Portfolio & Analysis Signals
  portfolio = signal<Portfolio | null>(null);
  monteCarloResult = signal<MonteCarloResult | null>(null);
  walkForwardAnalysis = signal<WalkForwardResult | null>(null);
  correlationMatrix = computed(() => {
      const strategies = this.selectedStrategies();
      if (strategies.length < 2) return null;

      const returnsMatrix = strategies.map(s => this.calculateReturns(s.equity));
      const minLength = Math.min(...returnsMatrix.map(r => r.length));
      if (minLength === 0) return null;

      const truncatedReturns = returnsMatrix.map(r => r.slice(0, minLength));
      const means = truncatedReturns.map(r => this.mean(r));
      const stdevs = truncatedReturns.map((r, i) => this.stdev(r, means[i]));

      const matrix: { x: string, y: string, value: number }[] = [];
      const labels = strategies.map(s => String(s.id));

      for (let i = 0; i < strategies.length; i++) {
        for (let j = 0; j < strategies.length; j++) {
          const corr = this.correlation(
            truncatedReturns[i],
            truncatedReturns[j],
            means[i],
            means[j],
            stdevs[i],
            stdevs[j]
          );
          matrix.push({ x: labels[j], y: labels[i], value: isNaN(corr) ? 1 : corr });
        }
      }
      return { matrix, labels };
    });
  
  // UI State Signals
  activeMetricsTab = signal<'equal' | 'montecarlo'>('equal');
  activePortfolioMetric = signal<AnalyzablePortfolioMetric | null>(null);
  strategyImpactValues = signal<Map<number, number>>(new Map());
  positiveMetrics: AnalyzablePortfolioMetric[] = ['sharpeRatio', 'calmarRatio', 'treynorRatio', 'avgProfitFactor', 'profitMode', 'profitPercentile80', 'recoveryFactor'];
  uploadStatus = signal<{ message: string; type: 'success' | 'error' } | null>(null);
  isTesting = signal(false);
  popupContent = signal<{ title: string; description: string } | null>(null);
  exportableFiles = signal<{fileName: string, content: string}[] | null>(null);
  detailedStrategy = signal<ProcessedStrategy | null>(null);
  
  // Constructor and Effects
  constructor() {
    let userDefaults = {};
    try {
        const storedDefaults = localStorage.getItem(USER_DEFAULTS_KEY);
        if (storedDefaults) {
            userDefaults = JSON.parse(storedDefaults);
        }
    } catch (e) {
        console.error("Failed to load user defaults from local storage", e);
    }
    this.effectiveDefaults = { ...this.baseDefaultFilters, ...userDefaults };
    this.filters.set(this.effectiveDefaults);
    
    effect(() => {
      const p = this.portfolio();
      if (p && this.portfolioChart()) {
        this.drawPortfolioChart(p.equityCurve);
      }
    }, { allowSignalWrites: true });

    effect(() => {
      const selected = this.selectedStrategies();
      if (selected.length > 0) {
        const newPortfolio = this.calculatePortfolio(selected);
        this.portfolio.set(newPortfolio);
        this.runMonteCarloSimulation(selected);
        this.runWalkForwardAnalysis(selected);
      } else {
        this.portfolio.set(null);
        this.monteCarloResult.set(null);
        this.walkForwardAnalysis.set(null);
        this.activePortfolioMetric.set(null);
      }
    }, { allowSignalWrites: true });

    effect(() => {
      const metric = this.activePortfolioMetric();
      const port = this.portfolio();
      if (metric && port) {
        this.calculateStrategyImpact(metric);
      } else {
        this.strategyImpactValues.set(new Map());
      }
    });

    effect(() => {
      const strategy = this.detailedStrategy();
      if (strategy && this.strategyEquityChart() && this.strategyDrawdownChart()) {
          this.drawStrategyEquityChart(strategy);
          this.drawStrategyDrawdownChart(strategy);
      }
    });

    effect(() => {
        const data = this.correlationMatrix();
        if (data && this.correlationHeatmap()) {
            this.drawCorrelationHeatmap(data.matrix, data.labels);
        }
    });

    effect(() => {
        const strategies = this.selectedStrategies();
        if (strategies.length > 1) {
            this.drawDdReturnsScatter(strategies);
            this.drawWinRateExpectancyScatter(strategies);
            this.drawRobustnessMitigationScatter(strategies);
        }
    });
  }

  // File Handling
  async onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    this.uploadStatus.set({ message: `Cargando ${input.files.length} archivos...`, type: 'success' });
    try {
        const strategiesMap = new Map<number, RawStrategy>();

        for (const file of Array.from(input.files)) {
            const content = await file.text();
            const data = JSON.parse(content);
            const strategiesFromFile: RawStrategy[] = Array.isArray(data) ? data : [data];
            
            for (const s of strategiesFromFile) {
                (s as any).originFile = file.name; 
                strategiesMap.set(s.magicNumber, s);
            }
        }
        
        const uniqueStrategies = Array.from(strategiesMap.values());
        
        this.rawStrategies.set(uniqueStrategies);
        this.uploadStatus.set({ message: `Carga exitosa: ${uniqueStrategies.length} estrategias únicas cargadas.`, type: 'success' });
    } catch (e) {
        this.uploadStatus.set({ message: 'Error al procesar el archivo JSON.', type: 'error' });
        console.error(e);
    }
  }

  // Strategy Processing
  private processStrategy(strat: RawStrategy): ProcessedStrategy {
    const trades: number[] = [];
    if (strat.equity && strat.equity.length > 1) {
      for (let i = 1; i < strat.equity.length; i++) {
        trades.push(strat.equity[i] - strat.equity[i-1]);
      }
    }
    
    const profitTrades = trades.filter(t => t > 0);
    const lossTrades = trades.filter(t => t < 0);
    
    const winRate = strat.backtestStats.countOfTrades > 0 ? (profitTrades.length / strat.backtestStats.countOfTrades) : 0;
    const avgWin = profitTrades.length > 0 ? this.mean(profitTrades) : 0;
    const avgLoss = lossTrades.length > 0 ? Math.abs(this.mean(lossTrades)) : 0;
    
    const expectancy = (winRate * avgWin) - ((1 - winRate) * avgLoss);
    const calmarRatio = strat.backtestStats.maxDrawdownPercent > 0 ? (this.calculateAnnualizedReturn(strat.equity) * 100) / strat.backtestStats.maxDrawdownPercent : 0;

    const profitMode = this.mode(profitTrades);
    const profitPercentile80 = this.percentile(profitTrades, 80);
    
    const robustnessScore = this.calculateRobustnessScore(strat.backtestStats, calmarRatio, expectancy);

    const meanTrade = this.mean(trades);
    const stdevTrade = this.stdev(trades, meanTrade);
    const zScore = stdevTrade > 0 ? (avgWin - meanTrade) / stdevTrade : 0;
    const iqr = this.percentile(trades, 75) - this.percentile(trades, 25);

    return {
      ...strat,
      calmarRatio,
      sortinoRatio: 0, 
      expectancy,
      winRate,
      robustnessScore,
      stabilityIndex: strat.backtestStats.rSquared,
      edgeQuality: strat.backtestStats.profitFactor * expectancy,
      profitMode,
      profitPercentile80,
      avgTradeDuration: strat.backtestStats.averagePositionLength,
      zScore,
      iqr,
      monkeyScore: 0,
      consistencyScore: 0,
    };
  }
  
  private calculateRobustnessScore(stats: BacktestStats, calmar: number, expectancy: number): number {
    const pfScore = Math.max(0, Math.min(20, (stats.profitFactor - 1) * 20));
    const rtdScore = Math.max(0, Math.min(20, (stats.returnToDrawdown - 1) * 5));
    const calmarScore = Math.max(0, Math.min(20, (calmar - 0.5) * 10));
    const sqnScore = Math.max(0, Math.min(15, (stats.systemQualityNumber - 1.5) * 10));
    const stabilityScore = Math.max(0, Math.min(15, (stats.rSquared / 100) * 15)); 
    const expectancyScore = expectancy > 0 && stats.profit > 0 && stats.countOfTrades > 0 ? Math.max(0, Math.min(10, (expectancy / (stats.profit / stats.countOfTrades)) * 20)) : 0;

    return pfScore + rtdScore + calmarScore + sqnScore + stabilityScore + expectancyScore;
  }

  // Portfolio Calculation
  private calculatePortfolio(strategies: ProcessedStrategy[]): Portfolio {
    if (strategies.length === 0) {
      throw new Error("Cannot calculate portfolio with no strategies");
    }
  
    const maxLength = Math.max(...strategies.map(s => s.equity.length));
    const portfolioEquityCurve = new Array(maxLength).fill(0);
    const initialCapital = strategies[0].initialAccount;
  
    portfolioEquityCurve[0] = initialCapital * strategies.length;
  
    for (let i = 1; i < maxLength; i++) {
      let dailyTotal = 0;
      for (const strat of strategies) {
        const equity = strat.equity;
        const prevValue = i > 0 && i - 1 < equity.length ? equity[i - 1] : initialCapital;
        const currentValue = i < equity.length ? equity[i] : prevValue;
        dailyTotal += currentValue - prevValue;
      }
      portfolioEquityCurve[i] = portfolioEquityCurve[i-1] + dailyTotal;
    }
    
    return this.calculatePortfolioMetrics(portfolioEquityCurve, strategies);
  }

  private calculatePortfolioMetrics(equityCurve: number[], strategies: ProcessedStrategy[]): Portfolio {
    const returns = this.calculateReturns(equityCurve);
    const totalProfit = equityCurve[equityCurve.length - 1] - equityCurve[0];
    
    const sharpeRatio = this.sharpeRatio(returns);
    const maxDrawdownStats = this.calculateMaxDrawdown(equityCurve);
    const annualizedReturn = this.calculateAnnualizedReturn(equityCurve);
    const calmarRatio = maxDrawdownStats.maxDrawdownPercent > 0 ? (annualizedReturn * 100) / maxDrawdownStats.maxDrawdownPercent : 0;
    
    const totalTrades = strategies.reduce((sum, s) => sum + s.backtestStats.countOfTrades, 0);
    const avgProfitFactor = this.mean(strategies.map(s => s.backtestStats.profitFactor));
    const avgWinRate = this.mean(strategies.map(s => s.winRate));

    const allProfitTrades = strategies.flatMap(s => {
      const trades: number[] = [];
      if (s.equity) {
        for (let i = 1; i < s.equity.length; i++) {
          trades.push(s.equity[i] - s.equity[i-1]);
        }
      }
      return trades.filter(t => t > 0);
    });

    const profitMode = this.mode(allProfitTrades);
    const profitPercentile80 = this.percentile(allProfitTrades, 80);

    const recoveryFactor = maxDrawdownStats.maxDrawdownValue > 0 ? totalProfit / maxDrawdownStats.maxDrawdownValue : 0;
    const maxConsecutiveLosses = this.calculateMaxConsecutiveLosses(equityCurve);
    
    const avgBeta = this.mean(strategies.map(s => s.beta ?? 1));
    const treynorRatio = avgBeta !== 0 ? annualizedReturn / avgBeta : 0;
    const rSquared = this.calculateRSquared(equityCurve);

    return {
      strategies: strategies,
      equityCurve: equityCurve,
      sharpeRatio,
      calmarRatio,
      treynorRatio,
      maxDrawdownPercent: maxDrawdownStats.maxDrawdownPercent,
      avgProfitFactor,
      avgWinRate,
      totalTrades,
      profitMode,
      profitPercentile80,
      recoveryFactor,
      maxConsecutiveLosses,
      rSquared,
    };
  }

  // Monte Carlo Simulation
  private runMonteCarloSimulation(strategies: ProcessedStrategy[], simulations = 250) {
    if (strategies.length === 0) {
      this.monteCarloResult.set(null);
      return;
    }

    const results: MonteCarloMetrics[] = [];
    
    const allTrades = strategies.map(s => {
      const trades = [];
      for (let i = 1; i < s.equity.length; i++) {
        trades.push(s.equity[i] - s.equity[i-1]);
      }
      return { id: s.id, initial: s.initialAccount, trades };
    });

    for (let i = 0; i < simulations; i++) {
      const simStrategies: any[] = allTrades.map(stratData => {
        const shuffledTrades = [...stratData.trades].sort(() => 0.5 - Math.random());
        const equity = [stratData.initial];
        for (const trade of shuffledTrades) {
          equity.push(equity[equity.length - 1] + trade);
        }
        return {
           equity,
           initialAccount: stratData.initial,
           backtestStats: { countOfTrades: 0, profitFactor: 0 },
           winRate: 0,
           beta: 1,
        };
      });

      const simPortfolio = this.calculatePortfolio(simStrategies as any);

      results.push({
        sharpeRatio: simPortfolio.sharpeRatio,
        calmarRatio: simPortfolio.calmarRatio,
        treynorRatio: simPortfolio.treynorRatio,
        maxDrawdownPercent: simPortfolio.maxDrawdownPercent,
        recoveryFactor: simPortfolio.recoveryFactor,
        profitMode: simPortfolio.profitMode,
        profitPercentile80: simPortfolio.profitPercentile80,
        maxConsecutiveLosses: simPortfolio.maxConsecutiveLosses,
        rSquared: simPortfolio.rSquared,
      });
    }

    const getPercentileMetrics = (p: number): MonteCarloMetrics => {
      const sorted = (key: keyof MonteCarloMetrics) => results.map(r => r[key]).sort((a, b) => a - b);
      return {
        sharpeRatio: this.percentile(sorted('sharpeRatio'), p),
        calmarRatio: this.percentile(sorted('calmarRatio'), p),
        treynorRatio: this.percentile(sorted('treynorRatio'), p),
        maxDrawdownPercent: this.percentile(sorted('maxDrawdownPercent'), p),
        recoveryFactor: this.percentile(sorted('recoveryFactor'), p),
        profitMode: this.percentile(sorted('profitMode'), p),
        profitPercentile80: this.percentile(sorted('profitPercentile80'), p),
        maxConsecutiveLosses: this.percentile(sorted('maxConsecutiveLosses'), p),
        rSquared: this.percentile(sorted('rSquared'), p),
      };
    };

    this.monteCarloResult.set({
      p5: getPercentileMetrics(5),
      p50: getPercentileMetrics(50),
      p95: getPercentileMetrics(95),
    });
  }
  
  // Selection Handlers
  toggleSelection(strategy: ProcessedStrategy, event: Event) {
    event.stopPropagation();
    this.selectedStrategiesMap.update(map => {
      const newMap = new Map(map);
      if (newMap.has(strategy.id)) {
        newMap.delete(strategy.id);
      } else {
        newMap.set(strategy.id, strategy);
      }
      return newMap;
    });
  }

  toggleSelectAll() {
    const areAllSelected = this.areAllFilteredSelected();
    const filtered = this.filteredStrategies();
    this.selectedStrategiesMap.update(map => {
        const newMap = new Map(map);
        if (areAllSelected) {
            filtered.forEach(s => newMap.delete(s.id));
        } else {
            filtered.forEach(s => newMap.set(s.id, s));
        }
        return newMap;
    });
  }

  isSelected(id: number): boolean {
    return this.selectedStrategiesMap().has(id);
  }

  reiniciarPortfolio() {
    this.selectedStrategiesMap.set(new Map());
  }

  exportPortfolio() {
    const selectedMagicNumbers = new Set(this.selectedStrategies().map(s => s.magicNumber));
    const allOriginalStrategies = this.rawStrategies();

    const strategiesByFile = new Map<string, RawStrategy[]>();
    for (const strat of allOriginalStrategies) {
        const originFile = (strat as any).originFile;
        if (!originFile) continue;

        if (!strategiesByFile.has(originFile)) {
            strategiesByFile.set(originFile, []);
        }
        strategiesByFile.get(originFile)!.push(strat);
    }
    
    const filesToExport: {fileName: string, content: string}[] = [];

    for (const [fileName, strategiesInFile] of strategiesByFile.entries()) {
        const portfolioStrategiesForFile = strategiesInFile
            .filter(s => selectedMagicNumbers.has(s.magicNumber))
            .map(s => {
                const { originFile, ...originalStrategy } = s as any;
                return originalStrategy;
            });

        if (portfolioStrategiesForFile.length > 0) {
            const fileContent = JSON.stringify(portfolioStrategiesForFile, null, 2);
            filesToExport.push({ fileName, content: fileContent });
        }
    }

    if (filesToExport.length > 0) {
        this.exportableFiles.set(filesToExport);
    }
  }

  downloadFile(fileName: string, content: string) {
    const blob = new Blob([content], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  closeExportModal() {
      this.exportableFiles.set(null);
  }

  // Filter and Sort Handlers
  updateFilter(key: keyof Filters, event: Event) {
    const value = (event.target as HTMLInputElement).value;
    this.filters.update(f => ({ ...f, [key]: Number(value) }));
  }

  saveFilterAsDefault(key: keyof Filters) {
    const currentValue = this.filters()[key];
    this.effectiveDefaults[key] = currentValue;

    const userDefaultsToSave: Partial<Filters> = {};
    for (const k in this.effectiveDefaults) {
        const filterKey = k as keyof Filters;
        if (this.effectiveDefaults[filterKey] !== this.baseDefaultFilters[filterKey]) {
            userDefaultsToSave[filterKey] = this.effectiveDefaults[filterKey];
        }
    }

    try {
        localStorage.setItem(USER_DEFAULTS_KEY, JSON.stringify(userDefaultsToSave));
    } catch (e) {
        console.error("Failed to save user defaults to local storage", e);
    }
  }

  reiniciarFiltros() {
    this.filters.set(this.effectiveDefaults);
  }

  setSort(key: SortKey) {
    this.sort.update(s => {
      if (s.key === key) {
        return { key, direction: s.direction === 'asc' ? 'desc' : 'asc' };
      }
      return { key, direction: 'desc' };
    });
  }

  // UI Handlers
  setActivePortfolioMetric(metric: AnalyzablePortfolioMetric) {
    this.activePortfolioMetric.update(current => current === metric ? null : metric);
  }
  
  showPopup(metric: string) {
    const content = this.getPopupContent(metric);
    this.popupContent.set(content);
  }

  hidePopup() {
    this.popupContent.set(null);
  }

  showStrategyDetails(strategy: ProcessedStrategy) {
    this.detailedStrategy.set(strategy);
  }

  hideStrategyDetails() {
    this.detailedStrategy.set(null);
  }
  
  private getPopupContent(metric: string): { title: string; description: string } {
    const definitions: { [key: string]: { title: string; description: string } } = {
      sharpeRatio: { title: 'Sharpe Ratio', description: 'Mide el retorno ajustado al riesgo. Un valor más alto indica un mejor rendimiento para la cantidad de riesgo asumido (volatilidad).' },
      calmarRatio: { title: 'Calmar Ratio', description: 'Mide el retorno en relación con el máximo drawdown. Un valor más alto es mejor, indicando una recuperación más rápida de las pérdidas.' },
      treynorRatio: { title: 'Treynor Ratio', description: 'Mide el retorno ajustado al riesgo sistemático (Beta). Es útil para evaluar cómo un activo o portafolio ha compensado a los inversores por asumir el riesgo del mercado.' },
      maxDrawdown: { title: 'Max Drawdown', description: 'La mayor caída porcentual desde un pico hasta un valle en el capital del portafolio. Mide el riesgo de pérdida máxima.' },
      recoveryFactor: { title: 'Factor de Recuperación', description: 'Beneficio Neto dividido por el Máximo Drawdown en valor monetario. Mide la capacidad de la estrategia para generar ganancias en relación con las pérdidas que ha sufrido.' },
      profitFactor: { title: 'Profit Factor', description: 'Ganancia bruta dividida por la pérdida bruta. Un valor superior a 1 indica rentabilidad.' },
      expectancy: { title: 'Expectativa (Expectancy)', description: 'Representa la ganancia o pérdida promedio que se puede esperar por cada operación. Se calcula como (Win Rate * Ganancia Promedio) - (Loss Rate * Pérdida Promedio). Un valor positivo indica una estrategia rentable a largo plazo.' },
      profitMode: { title: 'Moda del Beneficio', description: 'El tamaño de operación ganadora más frecuente. Un valor alto y positivo es deseable, ya que indica que las ganancias más comunes son significativas.' },
      profitPercentile80: { title: 'Percentil 80 del Beneficio', description: 'El valor por debajo del cual se encuentra el 80% de las operaciones ganadoras. Ayuda a entender la distribución de las ganancias y a ignorar valores atípicos extremos.' },
      avgTradeDuration: { title: 'Duración Promedio del Trade', description: 'El número promedio de barras/períodos que una operación permanece abierta.' },
      zScore: { title: 'Z-Score', description: 'Mide cuántas desviaciones estándar por encima o por debajo de la media está el beneficio de una operación ganadora promedio. Un Z-Score más alto indica que las operaciones ganadoras son significativamente más grandes que la operación promedio, lo que sugiere una ventaja robusta.' },
      iqr: { title: 'Rango Intercuartílico (IQR)', description: 'Es la diferencia entre el percentil 75 y el percentil 25 de los resultados de las operaciones. Un IQR más bajo indica una mayor consistencia en los resultados de las operaciones, con menos dispersión entre las ganancias y pérdidas.' },
      rSquared: { title: 'R-Cuadrado (R-Squared)', description: 'Mide la estabilidad y linealidad de la curva de capital. Un valor más alto (cercano a 100) indica que el crecimiento del capital es más consistente y predecible, similar a una línea recta ascendente.' },
      maxConsecutiveLosses: { title: 'Máx Pérdidas Consecutivas', description: 'El número máximo de períodos de trading (días, barras, etc.) consecutivos en los que el capital del portafolio disminuyó. Mide la duración de las rachas de pérdidas.' },
    };
    return definitions[metric] || { title: 'Métrica no encontrada', description: '' };
  }

  // Analysis runners
  calculateStrategyImpact(metric: AnalyzablePortfolioMetric) {
    const selected = this.selectedStrategies();
    if (selected.length < 2) {
      this.strategyImpactValues.set(new Map());
      return;
    }

    const newImpacts = new Map<number, number>();
    for (const strat of selected) {
      const portfolioWithout = selected.filter(s => s.id !== strat.id);
      const tempPortfolio = this.calculatePortfolio(portfolioWithout);
      newImpacts.set(strat.id, tempPortfolio[metric]);
    }
    this.strategyImpactValues.set(newImpacts);
  }
  
  runWalkForwardAnalysis(strategies: ProcessedStrategy[]) {
    if (strategies.length === 0) {
        this.walkForwardAnalysis.set(null);
        return;
    }
    const maxLength = Math.max(...strategies.map(s => s.equity.length));
    if (maxLength === 0) {
      this.walkForwardAnalysis.set(null);
      return;
    }
    const splitPoint = Math.floor(maxLength * 0.8);

    const getSegmentMetrics = (start: number, end: number): PortfolioSegmentMetrics => {
        const segmentStrategies = strategies.map(strat => ({
            ...strat,
            equity: strat.equity.slice(start, end)
        }));
        const portfolio = this.calculatePortfolio(segmentStrategies);
        return {
            sharpeRatio: portfolio.sharpeRatio,
            calmarRatio: portfolio.calmarRatio,
            maxDrawdownPercent: portfolio.maxDrawdownPercent,
            returns: portfolio.equityCurve.length > 1 ? (portfolio.equityCurve[portfolio.equityCurve.length - 1] / portfolio.equityCurve[0]) - 1 : 0,
        };
    };

    const inSample = getSegmentMetrics(0, splitPoint);
    const outOfSample = getSegmentMetrics(splitPoint, maxLength);

    const isRobust = 
        outOfSample.sharpeRatio > inSample.sharpeRatio * 0.5 &&
        outOfSample.calmarRatio > inSample.calmarRatio * 0.5 &&
        outOfSample.maxDrawdownPercent < inSample.maxDrawdownPercent * 1.5;

    this.walkForwardAnalysis.set({
        status: isRobust ? 'robust' : 'fragile',
        inSample,
        outOfSample
    });
  }

  runMassiveMonkeyTest() {
    this.isTesting.set(true);
    setTimeout(() => {
      this.isTesting.set(false);
    }, 3000);
  }

  clearAllStrategies() {
    this.rawStrategies.set([]);
    this.reiniciarPortfolio();
  }

  deleteStrategy(id: number) {
    this.rawStrategies.update(strategies => strategies.filter(s => s.id !== id));
    this.selectedStrategiesMap.update(map => {
      const newMap = new Map(map);
      if (newMap.has(id)) {
        newMap.delete(id);
      }
      return newMap;
    });
  }
  
  // D3 Charts
  private drawPortfolioChart(data: number[]) {
    const chartContainer = this.portfolioChart()?.nativeElement;
    if (!chartContainer || data.length < 2) return;

    d3.select(chartContainer).select('svg').remove();

    const margin = { top: 5, right: 0, bottom: 5, left: 0 };
    const width = chartContainer.clientWidth - margin.left - margin.right;
    const height = chartContainer.clientHeight - margin.top - margin.bottom;

    const svg = d3.select(chartContainer)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
      
    const x = d3.scaleLinear().range([0, width]);
    const y = d3.scaleLinear().range([height, 0]);

    x.domain([0, data.length - 1]);
    y.domain(d3.extent(data));

    const area = d3.area()
        .x((d: any, i: any) => x(i))
        .y0(height)
        .y1((d: any) => y(d))
        .curve(d3.curveMonotoneX);

    const line = d3.line()
      .x((d:any, i:any) => x(i))
      .y((d:any) => y(d))
      .curve(d3.curveMonotoneX);

    const defs = svg.append("defs");
    const gradient = defs.append("linearGradient")
        .attr("id", "portfolio-gradient")
        .attr("x1", "0%").attr("y1", "0%")
        .attr("x2", "0%").attr("y2", "100%");
    gradient.append("stop").attr("offset", "0%").attr("stop-color", "hsl(175, 80%, 30%)").attr("stop-opacity", 0.4);
    gradient.append("stop").attr("offset", "100%").attr("stop-color", "hsl(175, 80%, 30%)").attr("stop-opacity", 0);

    svg.append('path')
      .datum(data)
      .attr('fill', 'url(#portfolio-gradient)')
      .attr('d', area);

    svg.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', 'hsl(175, 80%, 45%)')
      .attr('stroke-width', 2)
      .attr('d', line);
  }

  private drawStrategyEquityChart(strategy: ProcessedStrategy) {
    const chartContainer = this.strategyEquityChart()?.nativeElement;
    const data = strategy.equity;
    if (!chartContainer || !data || data.length < 2) return;

    d3.select(chartContainer).select('svg').remove();

    const margin = { top: 10, right: 10, bottom: 20, left: 50 };
    const width = chartContainer.clientWidth - margin.left - margin.right;
    const height = chartContainer.clientHeight - margin.top - margin.bottom;

    const svg = d3.select(chartContainer)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
      
    const x = d3.scaleLinear().range([0, width]);
    const y = d3.scaleLinear().range([height, 0]);

    x.domain([0, data.length - 1]);
    y.domain(d3.extent(data));

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x).ticks(5))
        .attr("color", "#9ca3af");

    svg.append("g")
        .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format("$,.0f")))
         .attr("color", "#9ca3af");

    const line = d3.line()
      .x((d:any, i:any) => x(i))
      .y((d:any) => y(d))
      .curve(d3.curveMonotoneX);

    svg.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', 'hsl(175, 80%, 45%)')
      .attr('stroke-width', 2)
      .attr('d', line);
  }

  private drawStrategyDrawdownChart(strategy: ProcessedStrategy) {
    const chartContainer = this.strategyDrawdownChart()?.nativeElement;
    if (!chartContainer || !strategy.equity || strategy.equity.length < 2) return;

    const data = this.calculateDrawdownSeries(strategy.equity);
    d3.select(chartContainer).select('svg').remove();
    
    const margin = { top: 10, right: 10, bottom: 20, left: 50 };
    const width = chartContainer.clientWidth - margin.left - margin.right;
    const height = chartContainer.clientHeight - margin.top - margin.bottom;

    const svg = d3.select(chartContainer)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
      
    const x = d3.scaleLinear().range([0, width]);
    const y = d3.scaleLinear().range([0, height]); // Inverted for drawdown

    x.domain([0, data.length - 1]);
    y.domain([0, d3.max(data)]);

    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x).ticks(5))
        .attr("color", "#9ca3af");

    svg.append("g")
        .call(d3.axisLeft(y).ticks(5).tickFormat((d: any) => `${d}%`))
        .attr("color", "#9ca3af");
        
    const area = d3.area()
      .x((d: any, i: any) => x(i))
      .y0(0)
      .y1((d: any) => y(d))
      .curve(d3.curveMonotoneX);

    svg.append('path')
      .datum(data)
      .attr('fill', 'rgba(239, 68, 68, 0.5)')
      .attr('d', area);
  }

  private drawCorrelationHeatmap(data: {x: string, y: string, value: number}[], labels: string[]) {
    const chartContainer = this.correlationHeatmap()?.nativeElement;
    if (!chartContainer || !data) return;

    d3.select(chartContainer).select("svg").remove();

    const margin = { top: 30, right: 30, bottom: 30, left: 30 };
    const width = chartContainer.clientWidth - margin.left - margin.right;
    const height = chartContainer.clientHeight - margin.top - margin.bottom;

    const svg = d3.select(chartContainer)
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().range([0, width]).domain(labels).padding(0.05);
    svg.append("g")
        .style("font-size", "10px")
        .attr("transform", `translate(0, ${height})`)
        .call(d3.axisBottom(x).tickSize(0))
        .selectAll("text")
        .attr("transform", "translate(-10,10)rotate(-45)")
        .style("text-anchor", "end")
        .attr("fill", "#9ca3af");
    
    const y = d3.scaleBand().range([height, 0]).domain(labels).padding(0.05);
    svg.append("g")
        .style("font-size", "10px")
        .call(d3.axisLeft(y).tickSize(0))
        .select(".domain").remove()
        .selectAll("text")
        .attr("fill", "#9ca3af");

    const myColor = d3.scaleSequential().interpolator(d3.interpolateRdBu).domain([1, -1]);

    svg.selectAll()
        .data(data, (d: any) => d.x + ':' + d.y)
        .enter()
        .append("rect")
        .attr("x", (d: any) => x(d.x))
        .attr("y", (d: any) => y(d.y))
        .attr("rx", 4)
        .attr("ry", 4)
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .style("fill", (d: any) => myColor(d.value))
        .style("stroke-width", 4)
        .style("stroke", "none")
        .style("opacity", 0.8);
  }

  private drawDdReturnsScatter(strategies: ProcessedStrategy[]) {
    const chartContainer = this.ddReturnsScatter()?.nativeElement;
    if (!chartContainer || strategies.length < 2) return;

    const data = strategies.map(s => ({
        id: s.id,
        x: s.backtestStats.maxDrawdownPercent,
        y: this.calculateAnnualizedReturn(s.equity) * 100
    }));

    const xMode = this.mode(data.map(d => d.x));

    this.drawScatterPlot(
        chartContainer, 
        data, 
        'Max Drawdown (%)', 
        'Retorno Anualizado (%)',
        xMode
    );
  }

  private drawWinRateExpectancyScatter(strategies: ProcessedStrategy[]) {
    const chartContainer = this.winRateExpectancyScatter()?.nativeElement;
    if (!chartContainer || strategies.length < 2) return;

    const data = strategies.map(s => ({
        id: s.id,
        x: s.winRate * 100,
        y: s.expectancy
    }));
    
    const xMode = this.mode(data.map(d => d.x));

    this.drawScatterPlot(
        chartContainer, 
        data, 
        'Win Rate (%)', 
        'Expectativa ($)',
        xMode
    );
  }

  private drawRobustnessMitigationScatter(strategies: ProcessedStrategy[]) {
    const chartContainer = this.robustnessMitigationScatter()?.nativeElement;
    if (!chartContainer || strategies.length < 2) return;

    const data = strategies.map(s => ({
        id: s.id,
        x: s.robustnessScore,
        y: s.mitigationScore ?? 50
    }));

    const xMode = this.mode(data.map(d => d.x));

    this.drawScatterPlot(
        chartContainer, 
        data, 
        'Puntuación de Robustez', 
        'Puntuación de Mitigación',
        xMode
    );
  }

  private drawScatterPlot(
    container: HTMLElement, 
    data: { id: number; x: number; y: number }[], 
    xLabel: string, 
    yLabel: string,
    xMode?: number
  ) {
    d3.select(container).select('svg').remove();
    d3.select('body').select('.scatter-tooltip').remove();

    const tooltip = d3.select('body').append('div')
        .attr('class', 'scatter-tooltip')
        .style('position', 'absolute')
        .style('z-index', '50')
        .style('visibility', 'hidden')
        .style('background', 'rgba(17, 24, 39, 0.9)')
        .style('border', '1px solid #4b5563')
        .style('color', '#e5e7eb')
        .style('padding', '6px 10px')
        .style('border-radius', '6px')
        .style('font-size', '12px')
        .style('pointer-events', 'none');

    const margin = { top: 20, right: 20, bottom: 45, left: 55 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = container.clientHeight - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const xExtent = d3.extent(data, (d:any) => d.x);
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1 || 1;
    const x = d3.scaleLinear()
        .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
        .range([0, width]);

    svg.append('g')
        .attr('transform', `translate(0, ${height})`)
        .call(d3.axisBottom(x).ticks(5))
        .attr('color', '#6b7280');

    const yExtent = d3.extent(data, (d:any) => d.y);
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1 || 1;
    const y = d3.scaleLinear()
        .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
        .range([height, 0]);

    svg.append('g')
        .call(d3.axisLeft(y).ticks(5))
        .attr('color', '#6b7280');

    svg.append('text')
        .attr('text-anchor', 'middle')
        .attr('x', width / 2)
        .attr('y', height + margin.bottom - 5)
        .text(xLabel)
        .style('fill', '#9ca3af')
        .style('font-size', '11px');

    svg.append('text')
        .attr('text-anchor', 'middle')
        .attr('transform', 'rotate(-90)')
        .attr('y', -margin.left + 15)
        .attr('x', -height / 2)
        .text(yLabel)
        .style('fill', '#9ca3af')
        .style('font-size', '11px');
    
    if (xMode !== undefined && xMode >= (xExtent[0] - xPadding) && xMode <= (xExtent[1] + xPadding)) {
        svg.append('line')
            .attr('x1', x(xMode))
            .attr('y1', 0)
            .attr('x2', x(xMode))
            .attr('y2', height)
            .style('stroke', 'red')
            .style('stroke-width', 1.5)
            .style('stroke-dasharray', '4');
    }

    svg.append('g')
        .selectAll('dot')
        .data(data)
        .enter()
        .append('circle')
        .attr('cx', (d: any) => x(d.x))
        .attr('cy', (d: any) => y(d.y))
        .attr('r', 5)
        .style('fill', '#2dd4bf')
        .style('opacity', 0.7)
        .style('stroke', '#1f2937')
        .on('mouseover', (event: any, d: any) => {
            tooltip.style('visibility', 'visible').html(`ID: ${d.id}<br/>${xLabel.split(' ')[0]}: ${d.x.toFixed(2)}<br/>${yLabel.split(' ')[0]}: ${d.y.toFixed(2)}`);
        })
        .on('mousemove', (event: any) => {
            tooltip.style('top', (event.pageY - 10) + 'px').style('left', (event.pageX + 10) + 'px');
        })
        .on('mouseout', () => {
            tooltip.style('visibility', 'hidden');
        });
  }

  // Math Utilities
  private calculateReturns(equity: number[]): number[] {
    const returns: number[] = [];
    if (equity.length < 2) return [];
    for (let i = 1; i < equity.length; i++) {
        const prev = equity[i - 1] > 0 ? equity[i-1] : 1;
        returns.push((equity[i] / prev) - 1);
    }
    return returns;
  }

  private calculateDrawdownSeries(equity: number[]): number[] {
    if (equity.length < 2) return [];
    let peak = equity[0];
    const drawdownSeries: number[] = [0];

    for (let i = 1; i < equity.length; i++) {
        if (equity[i] > peak) {
            peak = equity[i];
        }
        const drawdown = (peak - equity[i]) / peak;
        drawdownSeries.push(drawdown * 100);
    }
    return drawdownSeries;
  }

  private mean(arr: number[]): number {
    return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  }
  
  private stdev(arr: number[], meanVal: number): number {
    if (arr.length < 2) return 0;
    const sqDiffs = arr.map(val => Math.pow(val - meanVal, 2));
    return Math.sqrt(this.mean(sqDiffs));
  }
  
  private variance(arr: number[], meanVal: number): number {
    if (arr.length < 2) return 0;
    return this.mean(arr.map(val => Math.pow(val - meanVal, 2)));
  }

  private covariance(arr1: number[], arr2: number[], mean1: number, mean2: number): number {
      let covar = 0;
      const len = Math.min(arr1.length, arr2.length);
      if(len === 0) return 0;
      for (let i = 0; i < len; i++) {
          covar += (arr1[i] - mean1) * (arr2[i] - mean2);
      }
      return covar / len;
  }

  private correlation(arr1: number[], arr2: number[], mean1: number, mean2: number, stdev1: number, stdev2: number): number {
      if (stdev1 === 0 || stdev2 === 0) return 1;
      const covariance = this.covariance(arr1, arr2, mean1, mean2);
      return covariance / (stdev1 * stdev2);
  }
  
  private calculateAnnualizedReturn(equity: number[], periodsPerYear = 252): number {
    if (equity.length < 2) return 0;
    const totalReturn = (equity[equity.length - 1] / equity[0]) - 1;
    const years = equity.length / periodsPerYear;
    return Math.pow(1 + totalReturn, 1 / years) - 1;
  }

  private sharpeRatio(returns: number[], riskFreeRate = 0, periodsPerYear = 252): number {
    if (returns.length < 2) return 0;
    const meanReturn = this.mean(returns);
    const stdDev = this.stdev(returns, meanReturn);
    if (stdDev === 0) return 0;
    return (meanReturn - riskFreeRate) / stdDev * Math.sqrt(periodsPerYear);
  }
  
  private calculateMaxDrawdown(equity: number[]): { maxDrawdownPercent: number, maxDrawdownValue: number } {
    if (equity.length < 2) return { maxDrawdownPercent: 0, maxDrawdownValue: 0 };
    let peak = equity[0];
    let maxDrawdown = 0;
    let maxDrawdownValue = 0;

    for (let i = 1; i < equity.length; i++) {
        if (equity[i] > peak) {
            peak = equity[i];
        }
        const drawdown = (peak - equity[i]) / peak;
        if (drawdown > maxDrawdown) {
            maxDrawdown = drawdown;
            maxDrawdownValue = peak - equity[i];
        }
    }
    return { maxDrawdownPercent: maxDrawdown * 100, maxDrawdownValue };
  }

  private calculateMaxConsecutiveLosses(equity: number[]): number {
    if (equity.length < 2) return 0;
    let maxLosingStreak = 0;
    let currentLosingStreak = 0;
    for (let i = 1; i < equity.length; i++) {
      if (equity[i] < equity[i - 1]) {
        currentLosingStreak++;
      } else {
        if (currentLosingStreak > maxLosingStreak) {
          maxLosingStreak = currentLosingStreak;
        }
        currentLosingStreak = 0;
      }
    }
    if (currentLosingStreak > maxLosingStreak) {
      maxLosingStreak = currentLosingStreak;
    }
    return maxLosingStreak;
  }

  private percentile(arr: number[], p: number): number {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) {
      return sorted[lower];
    }
    return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
  }
  
  private mode(arr: number[]): number {
    if (arr.length === 0) return 0;
    const counts = new Map<number, number>();
    let maxCount = 0;
    let mode = arr[0];
    
    for (const num of arr) {
      const rounded = Math.round(num * 100) / 100;
      const count = (counts.get(rounded) || 0) + 1;
      counts.set(rounded, count);
      if (count > maxCount) {
        maxCount = count;
        mode = rounded;
      }
    }
    return mode;
  }

  private calculateRSquared(data: number[]): number {
    if (data.length < 2) return 0;
    const n = data.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = data;

    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.map((xi, i) => xi * y[i]).reduce((a, b) => a + b, 0);
    const sumX2 = x.map(xi => xi * xi).reduce((a, b) => a + b, 0);
    const meanY = sumY / n;

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    let ssTot = 0;
    let ssRes = 0;

    for (let i = 0; i < n; i++) {
        const yHat = slope * x[i] + intercept;
        ssTot += Math.pow(y[i] - meanY, 2);
        ssRes += Math.pow(y[i] - yHat, 2);
    }

    if (ssTot === 0) return 100;
    
    return (1 - (ssRes / ssTot)) * 100;
  }
}