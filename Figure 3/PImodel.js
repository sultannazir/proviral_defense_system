Simulation = require('../cacatoo.js') // Loads the Simulation class from installation, or from local package like below

// Parse arguments
const yargs = require('yargs')
// const { hideBin } = require('yargs/helpers')
// var cmd_params = yargs(hideBin(process.argv)).argv
var cmd_params = yargs.argv

// Parameters

b = 0.01
c = 0.0
ind = 0.0
d = 0.002
beta = cmd_params.beta || 1.0
integration = cmd_params.lambda || 0.01
induction = cmd_params.sigma || 0.0001

r = 0.25
c_alpha = cmd_params.c_alpha || 0.8
seed = cmd_params.seed
latency = 20
gamma = 1/latency/5

muA = 0.01
step = 0.01

GR = 10**(-0.25*cmd_params.GR) || 0
GT = 10**(-0.25*cmd_params.GT) || 0

var size = 200

// for demo
var distance = 40
var d1 = Math.floor(distance/2)
var d2 = Math.floor(0.866*distance)

function histogram(A, binSize = 0.01) {
  const nBins = Math.round(1 / binSize); // 100

  var bins = new Array(nBins).fill(0);

  for (let i = 0; i < A.length; i++) {
    let x = A[i];

    if (x < 0 || x > 1) continue;

    let ix = Math.floor(x / binSize);

    // handle edge case where value === 1
    if (ix === nBins) ix = nBins - 1;

    bins[ix]++;
  }

  return bins;
}

function shuffle(a) {
    var j, x, i
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(sim.rng.genrand_real2() * (i + 1))
        x = a[i]
        a[i] = a[j]
        a[j] = x
    }
    return a
}

function gaussianRandom() {
    // mean 0, std dev 1
    let u1 = sim.rng.genrand_real1();
    let u2 = sim.rng.genrand_real1();

    // avoid log(0)
    u1 = Math.max(u1, Number.EPSILON);

    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function gaussianStep(A, D, min, max) {
    let step = D * gaussianRandom();
    let newA = A + step;

    // 3. truncate to [0, 1]
    return Math.min(max, Math.max(min, newA));
}

////////////////////////////////////

let config = {
    title: "PI model of virus coinfection",
    description: "Individual based model of hyperparasitism evolution",
    maxtime: 100000, // 1000000, demo: 50000
    seed: seed,
    skip: 0,
    scale: 2,
    ncol: size,
    nrow: size,
    wrap: [true, true],
    graph_interval: 1,
    fpsmeter: true,
    statecolours: { alive: {'empty':'black', 'S':'white', 'P':'yellow', 'I':'blue'}}
}

function burstsize(provirus_alpha, coinfection_alpha) {
    var burst1 = 1
    var burst2 = 1
    for (let t=0; t < latency; t++) {
        var avg_alpha = (provirus_alpha*burst1 + coinfection_alpha*burst2) / (burst1 + burst2)
        var off1 = burst1*r*(1-c_alpha*provirus_alpha)*avg_alpha
        var off2 = burst2*r*(1-c_alpha*coinfection_alpha)*avg_alpha
        burst1 += off1
        burst2 += off2
    }
    return [beta*gamma*burst1, beta*gamma*burst2]
}

sim = new Simulation(config)

sim.makeGridmodel("host")

sim.initialGrid(sim.host, "alive", 'empty', "S", 0.5, "P", 0.5) // evolution : S 0.5 P 0.5, demo: S 1.0 P 0.0
for (let i = 0; i < sim.host.nc; i++) for (let j = 0; j < sim.host.nr; j++){
    if (sim.host.grid[i][j].alive == "P") sim.host.grid[i][j].provirus = sim.rng.genrand_real1()
    else sim.host.grid[i][j].provirus = undefined
    if (sim.rng.genrand_real1() < 0.001) {  // demo: 0.0, evolution: 0.001
        sim.host.grid[i][j].alive = 'I'
        sim.host.grid[i][j].coinfection =sim.rng.genrand_real1()
        var bursts = burstsize(sim.host.grid[i][j].provirus, sim.host.grid[i][j].coinfection)
        sim.host.grid[i][j].infectivity1 = bursts[0]
        sim.host.grid[i][j].infectivity2 = bursts[1]
    }

    // Following for provirus survival demo
    
    // if ((i-size/2)**2 + (j-size/2)**2 < 100) {
    //     sim.host.grid[i][j].alive = 'I'
    //     sim.host.grid[i][j].coinfection = 0.6
    //     var bursts = burstsize(sim.host.grid[i][j].coinfection, sim.host.grid[i][j].coinfection)
    //     sim.host.grid[i][j].infectivity1 = 0
    //     sim.host.grid[i][j].infectivity2 = bursts[1]
    // }
    //
    // // (a, 0)
    // else if ((i-size/2 - distance)**2 + (j-size/2)**2 < 100) {
    //     sim.host.grid[i][j].alive = 'P'
    //     sim.host.grid[i][j].provirus = 0.0
    //     sim.host.grid[i][j].coinfection = undefined
    // }
    //
    // // (a/2, 0.866a)
    // else if ((i-size/2 - d1)**2 + (j-size/2 - d2)**2 < 100) {
    //   sim.host.grid[i][j].alive = 'P'
    //   sim.host.grid[i][j].provirus = 0.2
    //   sim.host.grid[i][j].coinfection = undefined
    // }
    //
    // // (-a/2, 0.866a)
    // else if ((i-size/2 + d1)**2 + (j-size/2 - d2)**2 < 100) {
    //   sim.host.grid[i][j].alive = 'P'
    //   sim.host.grid[i][j].provirus = 0.4
    //   sim.host.grid[i][j].coinfection = undefined
    // }
    //
    // // (-a, 0)
    // else if ((i-size/2 + distance)**2 + (j-size/2)**2 < 100) {
    //   sim.host.grid[i][j].alive = 'P'
    //   sim.host.grid[i][j].provirus = 0.6
    //   sim.host.grid[i][j].coinfection = undefined
    // }
    //
    // // (-a/2, -0.866a)
    // else if ((i-size/2 + d1)**2 + (j-size/2 + d2)**2 < 100) {
    //   sim.host.grid[i][j].alive = 'P'
    //   sim.host.grid[i][j].provirus = 0.8
    //   sim.host.grid[i][j].coinfection = undefined
    // }
    //
    // // (a/2, -0.866a)
    // else if ((i-size/2 - d1)**2 + (j-size/2 + d2)**2 < 100) {
    //   sim.host.grid[i][j].alive = 'P'
    //   sim.host.grid[i][j].provirus = 1.0
    //   sim.host.grid[i][j].coinfection = undefined
    // }

    else {
        sim.host.grid[i][j].coinfection = undefined
    }
}

sim.host.nextState = function (i, j) {

// If host is susceptible
if (this.grid[i][j].alive == "S") {
    var rand = sim.rng.genrand_real1()
    if (rand < GT) {
        var p = sim.rng.genrand_int(0,this.nc-1)
        var q = sim.rng.genrand_int(0,this.nr-1)
        nbr = this.grid[p][q]
    }
    else {
        var nbr = this.randomMoore8(this, i, j)
    }
    // Die with probability d
    var rand = sim.rng.genrand_real1()
    if (rand < d) {
        this.grid[i][j].alive = 'empty'
        this.grid[i][j].provirus = undefined
        this.grid[i][j].coinfection = undefined
    }
    else if (nbr.alive == 'I') {
        if (rand < d + beta*nbr.infectivity1) {
            if (sim.rng.genrand_real1() < integration) {
                this.grid[i][j].alive = 'P'
                if (sim.rng.genrand_real1() < muA) this.grid[i][j].provirus = gaussianStep(nbr.provirus, step, 0, 1)
                else this.grid[i][j].provirus = nbr.provirus
            }
            else {
                this.grid[i][j].alive = 'I'
                if (sim.rng.genrand_real1() < muA) this.grid[i][j].coinfection = gaussianStep(nbr.provirus, step, 0, 1)
                else this.grid[i][j].coinfection = nbr.provirus
                var bursts = burstsize(this.grid[i][j].coinfection, this.grid[i][j].coinfection)
                this.grid[i][j].infectivity1 = 0
                this.grid[i][j].infectivity2 = bursts[0]
            }
        }
        else if (rand < d + beta*nbr.infectivity1 + beta*nbr.infectivity2) {
            if (sim.rng.genrand_real1() < integration) {
                this.grid[i][j].alive = 'P'
                if (sim.rng.genrand_real1() < muA) this.grid[i][j].provirus = gaussianStep(nbr.coinfection, step, 0, 1)
                else this.grid[i][j].provirus = nbr.coinfection
            }
            else {
                this.grid[i][j].alive = 'I'
                if (sim.rng.genrand_real1() < muA) this.grid[i][j].coinfection = gaussianStep(nbr.coinfection, step, 0, 1)
                else this.grid[i][j].coinfection = nbr.coinfection
                var bursts = burstsize(this.grid[i][j].coinfection, this.grid[i][j].coinfection)
                this.grid[i][j].infectivity1 = 0
                this.grid[i][j].infectivity2 = bursts[0]
            }
        }
    }
}
else if (this.grid[i][j].alive == "P") {
    var rand = sim.rng.genrand_real1()
    if (rand < GT) {
        var p = sim.rng.genrand_int(0,this.nc-1)
        var q = sim.rng.genrand_int(0,this.nr-1)
        nbr = this.grid[p][q]
    }
    else {
        var nbr = this.randomMoore8(this, i, j)
    }
    // Die with probability d
    var rand = sim.rng.genrand_real1()
    if (d + induction + nbr.infectivity1 + nbr.infectivity2 > 1) console.log("Warning: probabilities sum to more than 1: "+d+" "+nbr.provirus+" "+nbr.coinfection)
    if (rand < d) {
        this.grid[i][j].alive = 'empty'
        this.grid[i][j].provirus = undefined
        this.grid[i][j].coinfection = undefined
    }
    else if (rand < d + induction) {
        this.grid[i][j].alive = 'I'
        var bursts = burstsize(this.grid[i][j].provirus, this.grid[i][j].provirus)
        this.grid[i][j].infectivity1 = bursts[0]
        this.grid[i][j].infectivity2 = 0
    }
    else if (nbr.alive == 'I') {
        if (rand < d + induction + beta*nbr.infectivity1) {
            this.grid[i][j].alive = 'I'
            if (sim.rng.genrand_real1() < muA) this.grid[i][j].coinfection = gaussianStep(nbr.provirus, step, 0, 1)
            else this.grid[i][j].coinfection = nbr.provirus
            var bursts = burstsize(this.grid[i][j].provirus, this.grid[i][j].coinfection)
            this.grid[i][j].infectivity1 = bursts[0]
            this.grid[i][j].infectivity2 = bursts[1]
        }
        else if (rand < d + induction + beta*nbr.infectivity1 + beta*nbr.infectivity2) {
            this.grid[i][j].alive = 'I'
            if (sim.rng.genrand_real1() < muA) this.grid[i][j].coinfection = gaussianStep(nbr.coinfection, step, 0, 1)
            else this.grid[i][j].coinfection = nbr.coinfection
            var bursts = burstsize(this.grid[i][j].provirus, this.grid[i][j].coinfection)
            this.grid[i][j].infectivity1 = bursts[0]
            this.grid[i][j].infectivity2 = bursts[1]
        }
    }
}

else if (this.grid[i][j].alive == 'I') {
    var rand = sim.rng.genrand_real1()
    if (rand < gamma) {
        this.grid[i][j].alive = 'empty'
        this.grid[i][j].provirus = undefined
        this.grid[i][j].coinfection = undefined
        this.grid[i][j].infectivity1 = undefined
        this.grid[i][j].infectivity2 = undefined
    }
}

else {
    var rand = sim.rng.genrand_real1()
    if (rand < GR) {
        var p = sim.rng.genrand_int(0,this.nc-1)
        var q = sim.rng.genrand_int(0,this.nr-1)
        nbr = this.grid[p][q]
    }
    else {
        var nbr = this.randomMoore8(this, i, j)
    }
    if (nbr.alive == 'S') {
        if (sim.rng.genrand_real1() < b) {
            this.grid[i][j].alive = 'S'
            this.grid[i][j].provirus = undefined
            this.grid[i][j].coinfection = undefined
        }
    }
    else if (nbr.alive == 'P') {
        if (sim.rng.genrand_real1() < b*(1-c)) {
            this.grid[i][j].alive = 'P'
            if (sim.rng.genrand_real1() < muA/2) this.grid[i][j].provirus = gaussianStep(nbr.provirus, step, 0, 1)
            else this.grid[i][j].provirus = nbr.provirus
            this.grid[i][j].coinfection = undefined
        }
    }

}

}

sim.host.update = function() {
    if(sim.mixhosts) sim.host.perfectMix()
    this.synchronous()

    //write abundance, evolution
    if (sim.time%1000==0) {
        var provirus_alpha = []
        var coinfection_alpha = []
        var numS = 0
        var numP = 0
        var numI = 0
        for (let i = 0; i < this.nc; i++) for (let j = 0; j < this.nr; j++){
            if (this.grid[i][j].alive == 'I') {
                coinfection_alpha.push(this.grid[i][j].coinfection)
                if (this.grid[i][j].provirus != undefined) coinfection_alpha.push(this.grid[i][j].provirus)
                numI++
            }
            else if (this.grid[i][j].alive == 'P') {
                provirus_alpha.push(this.grid[i][j].provirus)
                numP++
            }
            else if (this.grid[i][j].alive == 'S') {
                numS++
            }
        }
        PA = histogram(provirus_alpha)
        CA = histogram(coinfection_alpha)
        sim.write_append(JSON.stringify({seed: seed, GT: GT, GR: GR, time: sim.time, numS: numS, numP: numP, numI: numI, provirus: PA, free_virus: CA}) + '\n', 'GT_GR_sweep.txt')
        if (numS*numP*numI == 0) sim.stop()
    }

}


sim.start()
