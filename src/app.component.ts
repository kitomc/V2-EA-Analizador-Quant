import { ChangeDetectionStrategy, Component, signal, computed, ElementRef, viewChild, effect } from '@angular/core';
import { CommonModule } from '@angular/common';

import { RawStrategy, ProcessedStrategy, Filters, Portfolio, SortKey, SortDirection, BacktestStats } from './strategy.types';

declare var d3: any;

// FIX: Type aliases cannot be declared inside a class. Moved `AnalyzablePortfolioMetric` outside of AppComponent to fix compilation errors.
type AnalyzablePortfolioMetric =
  | 'sharpeRatio'
  | 'calmarRatio'
  | 'treynorRatio'
  | 'maxDrawdownPercent'
  | 'avgProfitFactor'
  | 'profitMode'
  | 'profitPercentile80';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {

  portfolioChart = viewChild<ElementRef>('portfolioChart');

  // Raw and Processed Data Signals
  rawStrategies = signal<RawStrategy[]>([]);
  baseProcessedStrategies = computed<ProcessedStrategy[]>(() => {
    return this.rawStrategies().map(strat => this.processStrategy(strat));
  });

  processedStrategies = computed<ProcessedStrategy[]>(() => {
    const strats = this.baseProcessedStrategies();
    
    if (strats.length < 2) {
      return strats.map(s => ({ ...s, treynorRatio: 0, beta: 1 }));
    }

    // 1. Get all returns
    const allReturns = strats.map(s => {
      const returns: number[] = [];
      for (let i = 1; i < s.equity.length; i++) {
        const prevEquity = s.equity[i-1] > 0 ? s.equity[i-1] : 1;
        returns.push((s.equity[i] / prevEquity) - 1);
      }
      return returns;
    });
    
    const minReturnLength = Math.min(...allReturns.map(r => r.length));

    if (minReturnLength === 0) {
        return strats.map(s => ({ ...s, treynorRatio: 0, beta: 1 }));
    }
    
    // 2. Calculate market returns (equal-weighted portfolio of all strategies)
    const marketReturns: number[] = [];
    for (let i = 0; i < minReturnLength; i++) {
      let sum = 0;
      for (const stratReturns of allReturns) {
        sum += stratReturns[i];
      }
      marketReturns.push(sum / allReturns.length);
    }
    
    // 3. Calculate variance of market returns
    const marketMean = marketReturns.reduce((a, b) => a + b, 0) / marketReturns.length;
    const marketVariance = marketReturns.reduce((acc, val) => acc + Math.pow(val - marketMean, 2), 0) / marketReturns.length;

    if (marketVariance === 0) {
      return strats.map(s => ({ ...s, treynorRatio: 0, beta: 1 }));
    }

    // 4. Calculate beta and Treynor for each strategy
    return strats.map((strat, index) => {
      const stratReturns = allReturns[index].slice(0, minReturnLength);
      const stratMean = stratReturns.reduce((a, b) => a + b, 0) / stratReturns.length;

      let covariance = 0;
      for (let i = 0; i < minReturnLength; i++) {
        covariance += (stratReturns[i] - stratMean) * (marketReturns[i] - marketMean);
      }
      covariance /= minReturnLength;

      const beta = covariance / marketVariance;
      
      const annualizedReturn = stratMean * 252; // Annualize assuming 252 periods/year
      
      const treynorRatio = beta !== 0 ? annualizedReturn / beta : 0;
      
      return { ...strat, treynorRatio, beta };
    });
  });

  // Filter and Sorting Signals
  defaultFilters: Filters = {
    minProfitFactor: 1.0,
    minCalmarRatio: 1,
    maxDrawdown: 30.0,
    minTrades: 100,
    minRobustnessScore: 0,
    minWinRate: 0.0,
    minProfitMode: 2.0,
    minProfitPercentile80: 2,
    maxAvgTradeDuration: 20,
    minExpectancy: 0.0,
  };
  filters = signal<Filters>({ ...this.defaultFilters });
  sort = signal<{ key: SortKey; direction: SortDirection }>({ key: 'robustnessScore', direction: 'desc' });

  // Computed Signals for Display
  filteredStrategies = computed(() => {
    const f = this.filters();
    return this.processedStrategies()
      .filter(s => 
        s.backtestStats.profitFactor >= f.minProfitFactor &&
        s.calmarRatio >= f.minCalmarRatio &&
        s.backtestStats.maxDrawdownPercent <= f.maxDrawdown &&
        s.backtestStats.countOfTrades >= f.minTrades &&
        s.robustnessScore >= f.minRobustnessScore &&
        s.winRate >= f.minWinRate &&
        s.profitMode >= f.minProfitMode &&
        s.profitPercentile80 >= f.minProfitPercentile80 &&
        s.avgTradeDuration <= f.maxAvgTradeDuration &&
        s.expectancy >= f.minExpectancy
      );
  });
  
  strategiesWithMitigation = computed<ProcessedStrategy[]>(() => {
    const strategies = this.filteredStrategies();
    const portfolio = this.portfolio();

    if (!portfolio || portfolio.strategies.length < 2) {
      return strategies.map(s => ({ ...s, mitigationScore: 50 })); // Neutral score
    }

    const portfolioDownDayIndices: number[] = [];
    for (let i = 1; i < portfolio.equityCurve.length; i++) {
        if (portfolio.equityCurve[i] < portfolio.equityCurve[i - 1]) {
            portfolioDownDayIndices.push(i);
        }
    }

    if (portfolioDownDayIndices.length === 0) {
        return strategies.map(s => ({ ...s, mitigationScore: 100 })); // Perfect score if no portfolio drawdowns
    }

    const performances = strategies.map(strat => {
        let totalReturnOnDownDays = 0;
        for (const index of portfolioDownDayIndices) {
            const prevEquity = strat.equity[index - 1] ?? strat.equity[0];
            const currentEquity = strat.equity[index] ?? strat.equity[strat.equity.length - 1];
            if (prevEquity > 0) {
                totalReturnOnDownDays += (currentEquity / prevEquity) - 1;
            }
        }
        const avgReturn = (totalReturnOnDownDays / portfolioDownDayIndices.length) * 100; // As percentage
        return { id: strat.id, performance: avgReturn };
    });

    const minPerf = Math.min(...performances.map(p => p.performance));
    const maxPerf = Math.max(...performances.map(p => p.performance));
    
    const scores = new Map<number, number>();
    performances.forEach(p => {
        const score = (maxPerf === minPerf) ? 50 : this.normalize(p.performance, minPerf, maxPerf, 0, 100);
        scores.set(p.id, score);
    });

    return strategies.map(s => ({
        ...s,
        mitigationScore: scores.get(s.id) ?? 0
    }));
  });

  sortedStrategies = computed(() => {
    return [...this.strategiesWithMitigation()].sort((a, b) => {
        const { key, direction } = this.sort();
        const valA = this.getSortableValue(a, key);
        const valB = this.getSortableValue(b, key);
        if (valA < valB) return direction === 'asc' ? -1 : 1;
        if (valA > valB) return direction === 'asc' ? 1 : -1;
        return 0;
    });
  });

  // Selection and Portfolio Signals
  selectedIds = signal<Set<number>>(new Set());
  selectedStrategies = computed<ProcessedStrategy[]>(() => {
    const ids = this.selectedIds();
    return this.processedStrategies().filter(s => ids.has(s.id));
  });
   areAllFilteredSelected = computed<boolean>(() => {
    const filtered = this.filteredStrategies();
    if (filtered.length === 0) {
      return false;
    }
    const ids = this.selectedIds();
    return filtered.every(s => ids.has(s.id));
  });
  portfolio = computed<Portfolio | null>(() => this.calculatePortfolio(this.selectedStrategies()));
  
  // UI Status
  uploadStatus = signal<{ type: 'success' | 'error'; message: string } | null>(null);
  monkeyTestStatus = signal<string | null>(null);
  isTesting = signal<boolean>(false);
  // FIX: Changed `this['AnalyzablePortfolioMetric']` to `AnalyzablePortfolioMetric` to correctly reference the type alias.
  activePortfolioMetric = signal<AnalyzablePortfolioMetric | null>(null);
  popupContent = signal<{ title: string; description: string } | null>(null);

  strategyMetricImpact = computed(() => {
    const metric = this.activePortfolioMetric();
    const currentPortfolio = this.portfolio();
    const strategies = this.selectedStrategies();
    const impactMap = new Map<number, 'improves' | 'worsens'>();

    if (!metric || !currentPortfolio || strategies.length < 2) {
        return impactMap;
    }

    const baselineValue = currentPortfolio[metric as keyof Portfolio] as number;
    
    // FIX: Changed `this['AnalyzablePortfolioMetric']` to `AnalyzablePortfolioMetric` to correctly reference the type alias and fix the assignment error.
    const positiveMetrics: AnalyzablePortfolioMetric[] = ['sharpeRatio', 'calmarRatio', 'treynorRatio', 'avgProfitFactor', 'profitMode', 'profitPercentile80'];
    const isPositive = positiveMetrics.includes(metric);

    strategies.forEach(strat => {
        const portfolioWithoutStrat = this.calculatePortfolio(strategies.filter(s => s.id !== strat.id));
        if (!portfolioWithoutStrat) return;

        const valueWithoutStrat = portfolioWithoutStrat[metric as keyof Portfolio] as number;

        if (isPositive) {
            if (baselineValue > valueWithoutStrat) {
                impactMap.set(strat.id, 'improves');
            } else if (baselineValue < valueWithoutStrat) {
                impactMap.set(strat.id, 'worsens');
            }
        } else { // Negative metric (maxDrawdownPercent)
            if (baselineValue < valueWithoutStrat) {
                impactMap.set(strat.id, 'improves');
            } else if (baselineValue > valueWithoutStrat) {
                impactMap.set(strat.id, 'worsens');
            }
        }
    });

    return impactMap;
  });

  private metricDefinitions: { [key: string]: { title: string; description: string } } = {
    robustnessScore: { 
      title: 'Puntaje de Robustez', 
      description: 'Una métrica compuesta (0-100) que agrega el Monkey Score, Calmar, Sortino y otras métricas de estabilidad. Un puntaje alto sugiere que el rendimiento de la estrategia es consistente, de alta calidad y no se debe al azar.' 
    },
    calmarRatio: { 
      title: 'Ratio Calmar', 
      description: 'Mide el rendimiento ajustado al riesgo, comparando el retorno anualizado con el máximo drawdown. Un ratio > 1 es bueno, > 3 es excelente. Indica cuántas unidades de retorno se obtienen por cada unidad de riesgo (drawdown).' 
    },
    maxDrawdown: { 
      title: 'Drawdown Máximo (%)', 
      description: 'La mayor caída porcentual desde un pico de equity hasta su punto más bajo. Es un indicador clave del riesgo y la posible pérdida que un inversor podría experimentar. Un drawdown más bajo es preferible.' 
    },
    profitFactor: { 
      title: 'Factor de Beneficio (Profit Factor)', 
      description: 'Calcula la relación entre la ganancia bruta y la pérdida bruta (Ganancias Totales / Pérdidas Totales). Un valor > 1 indica rentabilidad. Por ejemplo, un PF de 2 significa que las ganancias son el doble que las pérdidas.' 
    },
    expectancy: { 
      title: 'Expectativa Matemática', 
      description: 'La ganancia o pérdida promedio esperada por cada operación. Se calcula como (Tasa de Acierto × Ganancia Promedio) - (Tasa de Pérdida × Pérdida Promedio). Una expectativa positiva es crucial para la viabilidad a largo plazo.' 
    },
    profitMode: { 
      title: 'Moda del Beneficio', 
      description: 'La moda del beneficio por operación; es decir, el resultado de beneficio/pérdida más frecuente. Ayuda a entender el comportamiento típico de la estrategia, a diferencia del promedio que puede ser sesgado por outliers.' 
    },
    profitPercentile80: { 
      title: 'Percentil 80 del Beneficio (P80)', 
      description: 'Indica el valor por debajo del cual se encuentra el 80% de los beneficios de las operaciones. Es una medida robusta que ayuda a entender el rendimiento de la mayoría de los trades, ignorando los más extremos y atípicos.' 
    },
    minTrades: { 
      title: 'Operaciones Mínimas', 
      description: 'El número total de operaciones ejecutadas durante el backtest. Un número alto de operaciones (>100-200) proporciona mayor confianza estadística en los resultados de la estrategia.' 
    },
    avgTradeDuration: { 
      title: 'Duración Promedio del Trade', 
      description: 'El promedio de tiempo (en barras) que una operación permanece abierta. Ayuda a clasificar el estilo de la estrategia (scalping, intradía, swing) y a evaluar si se alinea con las condiciones del mercado y los costes.' 
    },
    sharpeRatio: { 
      title: 'Ratio de Sharpe', 
      description: 'Mide el retorno ajustado al riesgo en relación con la volatilidad (desviación estándar). Un ratio más alto indica un mejor rendimiento para la cantidad de riesgo asumido. Es ideal para comparar portafolios diversificados.' 
    },
    treynorRatio: { 
      title: 'Ratio de Treynor', 
      description: 'Mide el rendimiento ajustado al riesgo sistemático (Beta). Indica el exceso de retorno obtenido por cada unidad de riesgo de mercado asumido. Es útil para evaluar portafolios en el contexto del mercado general.' 
    }
  };

  constructor() {
    effect(() => {
      const p = this.portfolio();
      const chartEl = this.portfolioChart();
      
      if (!p && chartEl) {
        d3.select(chartEl.nativeElement).select('svg').remove();
        return;
      }

      if (p && chartEl) {
        this.drawPortfolioChart(chartEl.nativeElement, p.equityCurve);
      }
    });
  }

  // --- Data Processing Methods ---
  private calculateMode(numbers: number[]): number {
    if (numbers.length === 0) return 0;

    const frequency: { [key: number]: number } = {};
    let maxFreq = 0;
    let mode = numbers[0];

    for (const num of numbers) {
      const roundedNum = Math.round(num * 100) / 100;
      frequency[roundedNum] = (frequency[roundedNum] || 0) + 1;
      if (frequency[roundedNum] > maxFreq) {
        maxFreq = frequency[roundedNum];
        mode = roundedNum;
      }
    }
    return mode;
  }

  private calculatePercentile(numbers: number[], percentile: number): number {
    if (numbers.length === 0) return 0;
    
    const sorted = [...numbers].sort((a, b) => a - b);
    const index = (percentile / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    
    if (lower === upper) {
      return sorted[index];
    }
    
    if (lower < 0 || upper >= sorted.length) return 0;

    return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
  }

  private processStrategy(strat: RawStrategy): ProcessedStrategy {
    const stats = strat.backtestStats;
    const calmarRatio = stats.maxDrawdownPercent > 0 ? (stats.profit / strat.initialAccount) / (stats.maxDrawdownPercent / 100) : 0;
    
    const returns = strat.equity.slice(1).map((val, i) => (val / strat.equity[i]) - 1);
    const downsideReturns = returns.filter(r => r < 0);
    const downsideDeviation = Math.sqrt(downsideReturns.reduce((acc, val) => acc + val * val, 0) / (downsideReturns.length || 1));
    const avgReturn = returns.reduce((acc, val) => acc + val, 0) / (returns.length || 1);
    const sortinoRatio = downsideDeviation > 0 ? avgReturn / downsideDeviation * Math.sqrt(252) : 0; // Annualized

    const winRate = stats.winLossRatio / (1 + stats.winLossRatio);
    
    const grossProfit = stats.profitFactor > 1 ? (stats.profit * stats.profitFactor) / (stats.profitFactor - 1) : 0;
    const grossLoss = stats.profitFactor > 1 ? stats.profit / (stats.profitFactor - 1) : 0;
    const numLosses = stats.countOfTrades / (stats.winLossRatio + 1);
    const numWins = stats.countOfTrades - numLosses;
    const avgWin = numWins > 0 ? grossProfit / numWins : 0;
    const avgLoss = numLosses > 0 ? grossLoss / numLosses : 0;
    const expectancy = (winRate * avgWin) - ((1 - winRate) * avgLoss);

    const stabilityIndex = (winRate * stats.profitFactor) / (stats.maxConsecutiveLosses || 1);
    const edgeQuality = expectancy * Math.sqrt(stats.countOfTrades);
    
    const consistencyScore = this.normalize(stats.rSquared, 0, 100, 0, 100);
    const monkeyScore = this.calculateMonkeyScore(stats.rSquared, stats.systemQualityNumber, stats.countOfTrades);

    const robustnessScore = 
      0.3 * monkeyScore +
      0.2 * this.normalize(calmarRatio, 0, 5, 0, 100, true) +
      0.2 * this.normalize(sortinoRatio, 0, 3, 0, 100, true) +
      0.1 * consistencyScore +
      0.1 * this.normalize(edgeQuality, 0, 50, 0, 100, true) +
      0.1 * this.normalize(stabilityIndex, 0, 1, 0, 100, true);

    const tradeProfits = strat.equity.slice(1).map((val, i) => val - strat.equity[i]).filter(diff => diff !== 0);
    const profitMode = this.calculateMode(tradeProfits);
    const profitPercentile80 = this.calculatePercentile(tradeProfits, 80);

    return {
      ...strat,
      calmarRatio,
      sortinoRatio,
      expectancy,
      winRate,
      robustnessScore,
      stabilityIndex,
      edgeQuality,
      monkeyScore,
      consistencyScore,
      profitMode,
      profitPercentile80,
      avgTradeDuration: strat.backtestStats.averagePositionLength
    };
  }

  private calculateMonkeyScore(r2: number, sqn: number, trades: number): number {
    const r2Score = this.normalize(r2, 20, 80, 0, 100, true);
    const sqnScore = this.normalize(sqn, 1.5, 3.0, 0, 100, true);
    const tradesScore = this.normalize(trades, 100, 1000, 0, 100, true);
    return 0.4 * r2Score + 0.4 * sqnScore + 0.2 * tradesScore;
  }
  
  private normalize(value: number, min: number, max: number, newMin: number, newMax: number, clamp: boolean = false): number {
    if (value <= min) return clamp ? newMin : newMin;
    if (value >= max) return clamp ? newMax : newMax;
    return newMin + (value - min) * (newMax - newMin) / (max - min);
  }

  // --- UI Interaction Methods ---

  // FIX: Changed `this['AnalyzablePortfolioMetric']` to `AnalyzablePortfolioMetric` to correctly reference the type alias.
  setActivePortfolioMetric(metric: AnalyzablePortfolioMetric | null): void {
    if (!metric) return;
    
    if (this.activePortfolioMetric() === metric) {
      this.activePortfolioMetric.set(null); // Toggle off if clicked again
    } else {
      this.activePortfolioMetric.set(metric);
    }
  }

  async onFileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const files = input.files;
    if (!files || files.length === 0) {
      return;
    }
  
    const readFile = (file: File): Promise<RawStrategy[]> => {
      return new Promise((resolve, reject) => {
        if (file.type !== 'application/json') {
          return reject(new Error(`'${file.name}' no es un archivo JSON.`));
        }
  
        const reader = new FileReader();
        reader.onload = (e: ProgressEvent<FileReader>) => {
          try {
            const text = e.target?.result as string;
            if (!text) throw new Error(`El archivo '${file.name}' está vacío.`);
            
            const data = JSON.parse(text);
            if (!Array.isArray(data)) throw new Error(`'${file.name}' no contiene un array de estrategias.`);
            
            if (data.length > 0) {
                if (!('backtestStats' in data[0] && 'equity' in data[0])) {
                    throw new Error(`Las estrategias en '${file.name}' no tienen el formato esperado.`);
                }
                const originKey = `${data[0].symbol}-${data[0].period}`;
                const strategiesWithOrigin = data.map((strat: RawStrategy) => ({ ...strat, originKey }));
                resolve(strategiesWithOrigin);
            } else {
                resolve([]);
            }

          } catch (error: any) {
            reject(new Error(`Error al procesar '${file.name}': ${error.message}`));
          }
        };
  
        reader.onerror = () => {
          reject(new Error(`No se pudo leer el archivo '${file.name}'.`));
        };
  
        reader.readAsText(file);
      });
    };
  
    const promises = Array.from(files).map(readFile);
    const results = await Promise.allSettled(promises);
  
    const successfulStrategies: RawStrategy[] = [];
    let successCount = 0;
    let errorCount = 0;
  
    results.forEach(result => {
      if (result.status === 'fulfilled') {
        successfulStrategies.push(...result.value);
        successCount++;
      } else {
        console.error(result.reason);
        errorCount++;
      }
    });
  
    if (successfulStrategies.length > 0) {
      this.rawStrategies.set(successfulStrategies);
      this.selectedIds.set(new Set());
      let message = `Se cargaron ${successfulStrategies.length} estrategias de ${successCount} archivo(s).`;
      if (errorCount > 0) {
        message += ` Fallaron ${errorCount} archivo(s).`;
      }
      this.uploadStatus.set({ type: 'success', message });
    } else if (errorCount > 0) {
      this.uploadStatus.set({ type: 'error', message: `No se pudo cargar ninguna estrategia. ${errorCount} archivo(s) con errores.` });
    }
  
    input.value = '';
  }


  updateFilter(key: keyof Filters, event: Event) {
    const value = parseFloat((event.target as HTMLInputElement).value);
    this.filters.update(f => ({ ...f, [key]: value }));
  }

  reiniciarFiltros() {
    this.filters.set({ ...this.defaultFilters });
  }

  reiniciarPortfolio() {
    this.selectedIds.set(new Set());
    this.activePortfolioMetric.set(null);
  }

  setSort(key: SortKey) {
    this.sort.update(s => ({
      key,
      direction: s.key === key && s.direction === 'desc' ? 'asc' : 'desc'
    }));
  }

  toggleSelectAll() {
    const allSelected = this.areAllFilteredSelected();
    const filteredIds = this.filteredStrategies().map(s => s.id);

    this.selectedIds.update(currentIds => {
      const newIds = new Set(currentIds);
      if (allSelected) {
        // Deselect all that are currently filtered
        filteredIds.forEach(id => newIds.delete(id));
      } else {
        // Select all that are currently filtered
        filteredIds.forEach(id => newIds.add(id));
      }
      return newIds;
    });
  }

  toggleSelection(strat: ProcessedStrategy, event: Event) {
    event.stopPropagation();
    this.selectedIds.update(ids => {
      const newIds = new Set(ids);
      if (newIds.has(strat.id)) {
        newIds.delete(strat.id);
      } else {
        newIds.add(strat.id);
      }
      return newIds;
    });
  }
  
  isSelected(id: number): boolean {
    return this.selectedIds().has(id);
  }

  async runMassiveMonkeyTest(): Promise<void> {
    const originalStrategies = this.processedStrategies();
    const originalCount = originalStrategies.length;
    if (originalCount === 0) {
      return;
    }

    this.isTesting.set(true);
    this.uploadStatus.set(null);
    this.monkeyTestStatus.set(`Ejecutando test en ${originalCount} estrategias...`);
    
    await new Promise(resolve => setTimeout(resolve, 50));

    const ITERATIONS = 2500;
    const CONFIDENCE_LEVEL = 95;

    const robustStrategies = originalStrategies.filter(strat => {
      const tradeProfits = strat.equity.slice(1)
          .map((val, i) => val - strat.equity[i])
          .filter(diff => diff !== 0);
      
      if (tradeProfits.length < 2) {
          return false;
      }

      const randomFinalProfits: number[] = [];

      for (let i = 0; i < ITERATIONS; i++) {
          const shuffledProfits = [...tradeProfits];
          for (let j = shuffledProfits.length - 1; j > 0; j--) {
              const k = Math.floor(Math.random() * (j + 1));
              [shuffledProfits[j], shuffledProfits[k]] = [shuffledProfits[k], shuffledProfits[j]];
          }
          const randomFinalProfit = shuffledProfits.reduce((acc, p) => acc + p, 0);
          randomFinalProfits.push(randomFinalProfit);
      }

      const profitThreshold = this.calculatePercentile(randomFinalProfits, CONFIDENCE_LEVEL);
      const actualProfit = strat.backtestStats.profit;
      return actualProfit >= profitThreshold;
    });

    const robustCount = robustStrategies.length;
    const originalRaw = this.rawStrategies();
    const robustIds = new Set(robustStrategies.map(s => s.id));
    
    this.rawStrategies.set(originalRaw.filter(rs => robustIds.has(rs.id)));
    this.selectedIds.set(new Set());
    
    this.monkeyTestStatus.set(`Test completado: ${robustCount} de ${originalCount} estrategias`);
    this.isTesting.set(false);
  }
  
  exportPortfolio(): void {
    const selectedPortfolioStrategies = this.rawStrategies().filter(strat =>
        this.selectedIds().has(strat.id)
    );

    if (selectedPortfolioStrategies.length === 0) return;

    // Group strategies by their origin
    const strategiesByOrigin = new Map<string, RawStrategy[]>();
    for (const strat of selectedPortfolioStrategies) {
        if (strat.originKey) {
            if (!strategiesByOrigin.has(strat.originKey)) {
                strategiesByOrigin.set(strat.originKey, []);
            }
            // We can be sure it exists because we just set it.
            strategiesByOrigin.get(strat.originKey)!.push(strat);
        }
    }

    // Create and download a file for each origin group
    strategiesByOrigin.forEach((strategies, originKey) => {
        // Remove the temporary originKey before exporting to match original format
        const strategiesToExport = strategies.map(s => {
            const { originKey: _originKey, ...rest } = s;
            return rest;
        });

        const dataStr = JSON.stringify(strategiesToExport, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `portafolio_${originKey}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
  }


  clearAllStrategies(): void {
    if (this.rawStrategies().length > 0) {
      this.rawStrategies.set([]);
      this.selectedIds.set(new Set());
      this.uploadStatus.set({ type: 'success', message: 'Todas las estrategias han sido eliminadas.' });
      this.monkeyTestStatus.set(null);
    }
  }

  showPopup(metricKey: string): void {
    if (this.metricDefinitions[metricKey]) {
      this.popupContent.set(this.metricDefinitions[metricKey]);
    }
  }

  hidePopup(): void {
    this.popupContent.set(null);
  }

  // --- Portfolio Calculation ---

  private calculatePortfolio(strategies: ProcessedStrategy[]): Portfolio | null {
    if (strategies.length === 0) return null;

    const maxLength = Math.max(...strategies.map(s => s.equity.length));
    const portfolioEquityCurve: number[] = new Array(maxLength).fill(0);
    const initialAccount = strategies[0]?.initialAccount || 10000;
    
    // Equal Weighting
    const weight = 1 / strategies.length;

    for(let i = 0; i < maxLength; i++) {
        let periodValue = 0;
        for (const strat of strategies) {
            // Use last value if curve is shorter
            const value = strat.equity[i] ?? strat.equity[strat.equity.length - 1];
            periodValue += (value / initialAccount) * weight;
        }
        portfolioEquityCurve[i] = periodValue * initialAccount;
    }
    
    let maxEquity = initialAccount;
    let maxDd = 0;
    for (const equity of portfolioEquityCurve) {
        maxEquity = Math.max(maxEquity, equity);
        const drawdown = (maxEquity - equity) / maxEquity;
        maxDd = Math.max(maxDd, drawdown);
    }
    const maxDrawdownPercent = maxDd * 100;
    
    const finalProfit = (portfolioEquityCurve[portfolioEquityCurve.length - 1] - initialAccount);
    const calmarRatio = maxDrawdownPercent > 0 ? (finalProfit / initialAccount) / (maxDrawdownPercent / 100) : 0;

    const portfolioTradeProfits = portfolioEquityCurve.slice(1).map((val, i) => val - portfolioEquityCurve[i]).filter(diff => diff !== 0);
    const profitMode = this.calculateMode(portfolioTradeProfits);
    const profitPercentile80 = this.calculatePercentile(portfolioTradeProfits, 80);

    const portfolioBeta = strategies.reduce((sum, s) => sum + (s.beta ?? 1), 0) / strategies.length;
    const portfolioReturns: number[] = [];
    for (let i = 1; i < portfolioEquityCurve.length; i++) {
      const prev = portfolioEquityCurve[i-1];
      if (prev > 0) {
        portfolioReturns.push((portfolioEquityCurve[i] / prev) - 1);
      }
    }
    const avgPortfolioReturn = portfolioReturns.reduce((acc, val) => acc + val, 0) / (portfolioReturns.length || 1);
    const annualizedPortfolioReturn = avgPortfolioReturn * 252;
    const treynorRatio = portfolioBeta !== 0 ? annualizedPortfolioReturn / portfolioBeta : 0;

    return {
      strategies: strategies,
      equityCurve: portfolioEquityCurve,
      sharpeRatio: strategies.reduce((sum, s) => sum + s.backtestStats.sharpeRatio, 0) / strategies.length,
      calmarRatio: calmarRatio,
      treynorRatio: treynorRatio,
      maxDrawdownPercent: maxDrawdownPercent,
      avgProfitFactor: strategies.reduce((sum, s) => sum + s.backtestStats.profitFactor, 0) / strategies.length,
      avgWinRate: strategies.reduce((sum, s) => sum + s.winRate, 0) / strategies.length,
      totalTrades: strategies.reduce((sum, s) => sum + s.backtestStats.countOfTrades, 0),
      profitMode: profitMode,
      profitPercentile80: profitPercentile80,
    };
  }
  
  private getSortableValue(strat: ProcessedStrategy, key: SortKey) {
      if (key === 'mitigationScore') return strat.mitigationScore ?? -1;
      if (key === 'treynorRatio') return strat.treynorRatio ?? -Infinity;
      if (key in strat) return strat[key as keyof ProcessedStrategy];
      if (key in strat.backtestStats) return strat.backtestStats[key as keyof BacktestStats];
      return 0;
  }

  private drawPortfolioChart(element: HTMLElement, data: number[]): void {
    if (!element || data.length === 0) return;

    d3.select(element).select('svg').remove();

    const margin = { top: 10, right: 10, bottom: 20, left: 60 };
    const width = element.clientWidth - margin.left - margin.right;
    const height = element.clientHeight - margin.top - margin.bottom;

    if (width <= 0 || height <= 0) return;

    const svg = d3.select(element)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear()
      .domain([0, data.length - 1])
      .range([0, width]);

    const yDomain = d3.extent(data) as [number, number];
    const yPadding = (yDomain[1] - yDomain[0]) * 0.1;
    const y = d3.scaleLinear()
      .domain([yDomain[0] - yPadding, yDomain[1] + yPadding])
      .range([height, 0]);

    const xAxis = d3.axisBottom(x).ticks(Math.min(10, Math.floor(width/80))).tickSize(0).tickPadding(8);
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis)
      .select('.domain').remove();

    const yAxis = d3.axisLeft(y).ticks(5).tickSize(0).tickPadding(8).tickFormat(d3.format("~s"));
     svg.append('g')
      .call(yAxis)
      .select('.domain').remove();

    svg.selectAll('.tick text')
       .style('fill', '#9ca3af') // text-gray-400
       .style('font-size', '10px');

    const line = d3.line()
      .x((d: unknown, i: number) => x(i))
      .y((d: number) => y(d))
      .curve(d3.curveMonotoneX);
      
    const area = d3.area()
        .x((d: unknown, i: number) => x(i))
        .y0(height)
        .y1((d: number) => y(d))
        .curve(d3.curveMonotoneX);

    const defs = svg.append("defs");
    const gradient = defs.append("linearGradient")
        .attr("id", "area-gradient")
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("x1", 0).attr("y1", y(yDomain[0]))
        .attr("x2", 0).attr("y2", y(yDomain[1]));
    gradient.append("stop").attr("offset", "0%").attr("stop-color", "rgb(6 182 212 / 0)");
    gradient.append("stop").attr("offset", "100%").attr("stop-color", "rgb(6 182 212 / 0.3)");

    svg.append('path')
      .datum(data)
      .attr('fill', 'url(#area-gradient)')
      .attr('d', area);

    svg.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#06b6d4') // cyan-500
      .attr('stroke-width', 2)
      .attr('d', line);
  }

}