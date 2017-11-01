using System;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

public class PrinterNightmare
{
    public Variable<int> numExamples;

    // primary RVs
    public VariableArray<bool> Fuse;
    public VariableArray<bool> Drum;
    public VariableArray<bool> Toner;
    public VariableArray<bool> Paper;
    public VariableArray<bool> Roller;
    public VariableArray<bool> Burning;
    public VariableArray<bool> Quality;
    public VariableArray<bool> Wrinkled;
    public VariableArray<bool> MultPages;
    public VariableArray<bool> PaperJam;

    // RVs representing parameters of the distrib. of the primary RVs
    public Variable<double> ProbFuse;
    public Variable<double> ProbDrum;
    public Variable<double> ProbToner;
    public Variable<double> ProbPaper;
    public Variable<double> ProbRoller;
    public VariableArray<double> CPTBurning;
    public VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> CPTQuality;
    public VariableArray<VariableArray<double>, double[][]> CPTWrinkled;
    public VariableArray<VariableArray<double>, double[][]> CPTMultPages;
    public VariableArray<VariableArray<double>, double[][]> CPTPaperJam;

    // prior distrib. for the prob. and CPT variables
    public Variable<Beta> ProbFusePrior;
    public Variable<Beta> ProbDrumPrior;
    public Variable<Beta> ProbTonerPrior;
    public Variable<Beta> ProbPaperPrior;
    public Variable<Beta> ProbRollerPrior;
    public VariableArray<Beta> CPTBurningPrior;
    public VariableArray<VariableArray<VariableArray<Beta>, Beta[][]>, Beta[][][]> CPTQualityPrior;
    public VariableArray<VariableArray<Beta>, Beta[][]> CPTWrinkledPrior;
    public VariableArray<VariableArray<Beta>, Beta[][]> CPTMultPagesPrior;
    public VariableArray<VariableArray<Beta>, Beta[][]> CPTPaperJamPrior;

    // posterior distributions for the probability and CPT variables
    public Beta ProbFusePosterior;
    public Beta ProbDrumPosterior;
    public Beta ProbTonerPosterior;
    public Beta ProbPaperPosterior;
    public Beta ProbRollerPosterior;
    public Beta[] CPTBurningPosterior;
    public Beta[][][] CPTQualityPosterior;
    public Beta[][] CPTWrinkledPosterior;
    public Beta[][] CPTMultPagesPosterior;
    public Beta[][] CPTPaperJamPosterior;

    public InferenceEngine Engine = new InferenceEngine();

    public PrinterNightmare()
	{
        numExamples = Variable.New<int>().Named("numExample");
        Range nRange = new Range(numExamples);

        Range fuseRange = new Range(2).Named("fuseRange");
        Range drumRange = new Range(2).Named("drumRange");
        Range tonerRange = new Range(2).Named("tonerRange");
        Range paperRange = new Range(2).Named("paperRange");
        Range rollerRange = new Range(2).Named("rollerRange");
        Range burningRange = new Range(2).Named("burningRange");
        Range qualityRange = new Range(2).Named("qualityRange");
        Range wrinkledRange = new Range(2).Named("wrinkledRange");
        Range multPagesRange = new Range(2).Named("multPagesRange");
        Range paperJamRange = new Range(2).Named("paperJamRange");

        //
        // define priors and the parameters
        //

        ProbFusePrior = Variable.New<Beta>().Named("ProbFusePrior");
        ProbFuse = Variable<double>.Random(ProbFusePrior).Named("ProbFuse");
        ProbFuse.SetValueRange(fuseRange);

        ProbDrumPrior = Variable.New<Beta>().Named("ProbDrumPrior");
        ProbDrum = Variable<double>.Random(ProbDrumPrior).Named("ProbDrum");
        ProbDrum.SetValueRange(drumRange);

        ProbTonerPrior = Variable.New<Beta>().Named("ProbTonerPrior");
        ProbToner = Variable<double>.Random(ProbTonerPrior).Named("ProbToner");
        ProbToner.SetValueRange(tonerRange);

        ProbPaperPrior = Variable.New<Beta>().Named("ProbPaperPrior");
        ProbPaper = Variable<double>.Random(ProbPaperPrior).Named("ProbPaper");
        ProbPaper.SetValueRange(paperRange);

        ProbRollerPrior = Variable.New<Beta>().Named("ProbRollerPrior");
        ProbRoller = Variable<double>.Random(ProbRollerPrior).Named("Probroller");
        ProbRoller.SetValueRange(rollerRange);

        CPTBurningPrior = Variable.Array<Beta>(fuseRange).Named("ProbBurningPrior");
        CPTBurning = Variable.Array<double>(fuseRange).Named("CPTBurning");
        CPTBurning[fuseRange] = Variable<double>.Random(CPTBurningPrior[fuseRange]);
        CPTBurning.SetValueRange(burningRange);

        CPTQualityPrior = Variable.Array(Variable.Array(Variable.Array<Beta>(drumRange), tonerRange), paperRange).Named("ProbQualityPrior");
        CPTQuality = Variable.Array(Variable.Array(Variable.Array<double>(drumRange), tonerRange), paperRange).Named("CPTQuality");
        CPTQuality[paperRange][tonerRange][drumRange] = Variable<double>.Random(CPTQualityPrior[paperRange][tonerRange][drumRange]);
        CPTQuality.SetValueRange(qualityRange);

        CPTWrinkledPrior = Variable.Array(Variable.Array<Beta>(fuseRange), paperRange).Named("ProbWrinkledPrior");
        CPTWrinkled = Variable.Array(Variable.Array<double>(fuseRange), paperRange).Named("CPTWrinkled");
        CPTWrinkled[paperRange][fuseRange] = Variable<double>.Random(CPTWrinkledPrior[paperRange][fuseRange]);
        CPTWrinkled.SetValueRange(wrinkledRange);

        CPTMultPagesPrior = Variable.Array(Variable.Array<Beta>(rollerRange), paperRange).Named("ProbMultPagesPrior");
        CPTMultPages = Variable.Array(Variable.Array<double>(rollerRange), paperRange).Named("CPTMultPages");
        CPTMultPages[paperRange][rollerRange] = Variable<double>.Random(CPTMultPagesPrior[paperRange][rollerRange]);
        CPTMultPages.SetValueRange(multPagesRange);

        CPTPaperJamPrior = Variable.Array(Variable.Array<Beta>(rollerRange), fuseRange).Named("ProbPaperJamPrior");
        CPTPaperJam = Variable.Array(Variable.Array<double>(rollerRange), fuseRange).Named("CPTPaperJam");
        CPTPaperJam[fuseRange][rollerRange] = Variable<double>.Random(CPTPaperJamPrior[fuseRange][rollerRange]);
        CPTPaperJam.SetValueRange(paperJamRange);

        // define primary RVs
        Fuse = Variable.Array<bool>(nRange).Named("Fuse");
        Fuse[nRange] = Variable.Bernoulli(ProbFuse).ForEach(nRange);

        Drum = Variable.Array<bool>(nRange).Named("Drum");
        Drum[nRange] = Variable.Bernoulli(ProbDrum).ForEach(nRange);

        Toner = Variable.Array<bool>(nRange).Named("Toner");
        Toner[nRange] = Variable.Bernoulli(ProbToner).ForEach(nRange);

        Paper = Variable.Array<bool>(nRange).Named("Paper");
        Paper[nRange] = Variable.Bernoulli(ProbPaper).ForEach(nRange);

        Roller = Variable.Array<bool>(nRange).Named("Roller");
        Roller[nRange] = Variable.Bernoulli(ProbRoller).ForEach(nRange);

        Burning = AddChildFromOneParent(Fuse, CPTBurning).Named("Burning");
        Quality = AddChildFromThreeParents(Drum, Toner, Paper, CPTQuality).Named("Quality");
        Wrinkled = AddChildFromTwoParents(Fuse, Paper, CPTWrinkled).Named("Wrinkled");
        MultPages = AddChildFromTwoParents(Paper, Roller, CPTMultPages).Named("MultPages");
        PaperJam = AddChildFromTwoParents(Fuse, Roller, CPTPaperJam).Named("PaperJam");
    }

    public void LearnParameters(
            bool[] fuse, bool[] drum, bool[] toner, bool[] paper, bool[] roller,
            bool[] burning, bool[] quality, bool[] wrinkled, bool[] multPages, bool[] paperJam
        )
    {
        // set number of examples at runtime;
        // assuming all data arrays are of the same length
        numExamples.ObservedValue = fuse.Length;

        // set data
        Fuse.ObservedValue = fuse;
        Drum.ObservedValue = drum;
        Toner.ObservedValue = toner;
        Paper.ObservedValue = paper;
        Roller.ObservedValue = roller;
        Burning.ObservedValue = burning;
        Quality.ObservedValue = quality;
        Wrinkled.ObservedValue = wrinkled;
        MultPages.ObservedValue = multPages;
        PaperJam.ObservedValue = paperJam;

        // set uniform priors
        ProbFusePrior.ObservedValue = Beta.Uniform();
        ProbDrumPrior.ObservedValue = Beta.Uniform();
        ProbTonerPrior.ObservedValue = Beta.Uniform();
        ProbPaperPrior.ObservedValue = Beta.Uniform();
        ProbRollerPrior.ObservedValue = Beta.Uniform();
        CPTBurningPrior.ObservedValue = Enumerable.Repeat(Beta.Uniform(), 2).ToArray();
        CPTQualityPrior.ObservedValue = Enumerable.Repeat(Enumerable.Repeat(Enumerable.Repeat(Beta.Uniform(), 2).ToArray(), 2).ToArray(), 2).ToArray();
        CPTWrinkledPrior.ObservedValue = Enumerable.Repeat(Enumerable.Repeat(Beta.Uniform(), 2).ToArray(), 2).ToArray();
        CPTMultPagesPrior.ObservedValue = Enumerable.Repeat(Enumerable.Repeat(Beta.Uniform(), 2).ToArray(), 2).ToArray();
        CPTPaperJamPrior.ObservedValue = Enumerable.Repeat(Enumerable.Repeat(Beta.Uniform(), 2).ToArray(), 2).ToArray();
 
        // inference
        ProbFusePosterior = Engine.Infer<Beta>(ProbFuse);
        ProbDrumPosterior = Engine.Infer<Beta>(ProbDrum);
        ProbTonerPosterior = Engine.Infer<Beta>(ProbToner);
        ProbPaperPosterior = Engine.Infer<Beta>(ProbPaper);
        ProbRollerPosterior = Engine.Infer<Beta>(ProbRoller);
        CPTBurningPosterior = Engine.Infer<Beta[]>(CPTBurning);
        CPTQualityPosterior = Engine.Infer<Beta[][][]>(CPTQuality);
        CPTWrinkledPosterior = Engine.Infer<Beta[][]>(CPTWrinkled);
        CPTMultPagesPosterior = Engine.Infer<Beta[][]>(CPTMultPages);
        CPTPaperJamPosterior = Engine.Infer<Beta[][]>(CPTPaperJam);
    }

    public double QueryProbFuse(
        bool? burning,
        bool? quality,
        bool? wrinkled,
        bool? multPages,
        bool? paperJam,
        Beta probFusePrior,
        Beta probDrumPrior,
        Beta probTonerPrior,
        Beta probPaperPrior,
        Beta probRollerPrior,
        Beta[] cptBurningPrior,
        Beta[][][] cptQualityPrior,
        Beta[][] cptWrinkledPrior,
        Beta[][] cptMultPagesPrior,
        Beta[][] cptPaperJamPrior)
    {
        // reset observed data
        Fuse.ClearObservedValue();
        Drum.ClearObservedValue();
        Toner.ClearObservedValue();
        Paper.ClearObservedValue();
        Roller.ClearObservedValue();

        // only one issue description is given
        numExamples.ObservedValue = 1;

        // issue description
        if (burning.HasValue)
            this.Burning.ObservedValue = new bool[] { burning.Value };
        else
            this.Burning.ClearObservedValue();
        if (quality.HasValue)
            this.Quality.ObservedValue = new bool[] { quality.Value };
        else
            this.Quality.ClearObservedValue();
        if (wrinkled.HasValue)
            this.Wrinkled.ObservedValue = new bool[] { wrinkled.Value };
        else
            this.Wrinkled.ClearObservedValue();
        if (multPages.HasValue)
            this.MultPages.ObservedValue = new bool[] { multPages.Value };
        else
            this.MultPages.ClearObservedValue();
        if (paperJam.HasValue)
            this.PaperJam.ObservedValue = new bool[] { paperJam.Value };
        else
            this.PaperJam.ClearObservedValue();

        // set model learned parameters from a dataset of previous issues
        this.ProbFusePrior.ObservedValue = probFusePrior;
        this.ProbDrumPrior.ObservedValue = probDrumPrior;
        this.ProbTonerPrior.ObservedValue = probTonerPrior;
        this.ProbPaperPrior.ObservedValue = probPaperPrior;
        this.ProbRollerPrior.ObservedValue = probRollerPrior;
        this.CPTBurningPrior.ObservedValue = cptBurningPrior;
        this.CPTQualityPrior.ObservedValue = cptQualityPrior;
        this.CPTWrinkledPrior.ObservedValue = cptWrinkledPrior;
        this.CPTMultPagesPrior.ObservedValue = cptMultPagesPrior;
        this.CPTPaperJamPrior.ObservedValue = cptPaperJamPrior;

        // infer Fuse RV (array of 1 element)
        var fusePosterior = Engine.Infer<Bernoulli[]>(Fuse);
        return fusePosterior[0].GetProbTrue();
    }

    public static VariableArray<bool> AddChildFromOneParent(VariableArray<bool> parent, VariableArray<double> cpt)
    {
        var n = parent.Range;
        var child = Variable.Array<bool>(n);
        using (Variable.ForEach(n))
        {
            using (Variable.If(parent[n]))
                child[n] = Variable.Bernoulli(cpt[0]);
            using (Variable.IfNot(parent[n]))
                child[n] = Variable.Bernoulli(cpt[1]);
        }
        return child;
    }

    public static VariableArray<bool> AddChildFromTwoParents(VariableArray<bool> parent1, VariableArray<bool> parent2, VariableArray<VariableArray<double>, double[][]> cpt)
    {
        var n = parent1.Range;
        var child = Variable.Array<bool>(n);
        using (Variable.ForEach(n))
        {
            using (Variable.If(parent1[n]))
            {
                using (Variable.If(parent2[n]))
                    child[n] = Variable.Bernoulli(cpt[0][0]);
                using (Variable.IfNot(parent2[n]))
                    child[n] = Variable.Bernoulli(cpt[1][0]);
            }
            using (Variable.IfNot(parent1[n]))
            {
                using (Variable.If(parent2[n]))
                    child[n] = Variable.Bernoulli(cpt[0][1]);
                using (Variable.IfNot(parent2[n]))
                    child[n] = Variable.Bernoulli(cpt[1][1]);
            }
        }
        return child;
    }

    public static VariableArray<bool> AddChildFromThreeParents(VariableArray<bool> parent1, VariableArray<bool> parent2, VariableArray<bool> parent3, VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> cpt)
    {
        var n = parent1.Range;
        var child = Variable.Array<bool>(n);
        using (Variable.ForEach(n))
        {
            var parent2n = parent2[n] & Variable.Bernoulli(1.0);        // bug workaround
            using (Variable.If(parent1[n]))
            {
                using (Variable.If(parent2n))
                {
                    using (Variable.If(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[0][0][0]);
                    using (Variable.IfNot(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[1][0][0]);
                }
                using (Variable.IfNot(parent2n))
                {
                    using (Variable.If(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[0][1][0]);
                    using (Variable.IfNot(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[1][1][0]);
                }
            }
            using (Variable.IfNot(parent1[n]))
            {
                using (Variable.If(parent2n))
                {
                    using (Variable.If(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[0][0][1]);
                    using (Variable.IfNot(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[1][0][1]);
                }
                using (Variable.IfNot(parent2n))
                {
                    using (Variable.If(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[0][1][1]);
                    using (Variable.IfNot(parent3[n]))
                        child[n] = Variable.Bernoulli(cpt[1][1][1]);
                }
            }
        }
        return child;
    }

    public static bool[][] GetData()
    {
        bool[][] data = new bool[][]
        {
            new bool[] { false, false, false, true, false, false, false, false, false, false, false, false, true, false, true },    // fuse assembly malfunction
            new bool[] { false, false, false, false, true, false, false, true, false, false, true, true, false, false, false },     // drum unit
            new bool[] { true, true, false, false, false, true, false, true, false, false, false, true, false, false, false },      // toner out
            new bool[] { true, false, true, false, true, false, true, false, true, true, false, true, true, false, false },         // poor paper quality
            new bool[] { false, false, false, false, false, false, true, false, false, false, false, false, false, true, true },    // worn roller
            new bool[] { false, false, false, true, false, false, false, false, false, false, false, false, true, false, false },   // burning smell
            new bool[] { true, true, true, false, true, true, false, true, false, false, true, true, false, false, false },         // poor print quality
            new bool[] { false, false, true, false, false, false, false, false, true, false, false, false, true, true, true },      // wrinkled pages
            new bool[] { false, false, true, false, false, false, true, false, true, false, false, false, false, false, true },     // multiple pages fed
            new bool[] { false, false, true, true, false, false, true, true, true, true, false, false, false, true, false }          // paper jam
        };
        return data;
    }

    public static void Main()
    {
        Rand.Restart(2017);

        PrinterNightmare model = new PrinterNightmare();

        Console.WriteLine("\n******************************************************");
        Console.WriteLine("Step 1: Learning parameters from data (uniform priors)");
        Console.WriteLine("******************************************************");

        bool[][] data = GetData();

        try
        {
            // learn model parameters from data with uniform priors
            model.LearnParameters(data[0], data[1], data[2], data[3], data[4],
                              data[5], data[6], data[7], data[8], data[9]);
        }
        catch(NullReferenceException ex)
        {
            Console.WriteLine(ex);
        }

        Console.WriteLine("\n******************************************************");
        Console.WriteLine("Step 2: Query the model for P(Fuse | Burn, Paper Jam)");
        Console.WriteLine("******************************************************");

        double probFuseGivenBurningAndPaperJam = model.QueryProbFuse(
            true, false, false, false, true,    // set an issue indicator to null if it is unobserved (unknown)
            model.ProbFusePosterior, model.ProbDrumPosterior, model.ProbTonerPosterior, model.ProbPaperPosterior, model.ProbRollerPosterior,
            model.CPTBurningPosterior, model.CPTQualityPosterior, model.CPTWrinkledPosterior, model.CPTMultPagesPosterior, model.CPTPaperJamPosterior);

        Console.WriteLine("P(Fuse = True | Burn, Paper Jam) = {0:0.00}", probFuseGivenBurningAndPaperJam);

        Console.ReadKey();
    }
}
