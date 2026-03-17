# Syntax

Runs a fully connected Multi Layer Perceptron between a set of inputs and outputs.

```
fix ID group-ID multiLayerPerceptron nLayers nNeuronsPerLayer inputList outputList activationFunctions decimate
```

- **nLayers** : Number of hidden layers >= 1
- **nNeuronsPerLayer** : Number of neurons per hidden layer >= 1
- **inputList** : List of semicolon separated input properties chosen among the supported values below.
- **outputList** : List of semicolon separated output properties chosen among the supported values below.
- **activationFunctions** : List of semicolon separated activation functions chosen among the supported values below. The number of values must match the number of hidden layers + 1 for the outputs. The last value can either be a single activation function, or per-output activation if written in brackets and comma separated.
- **decimate** : Runs the fix only every _decimate_ timesteps

Examples :
```
fix 1 all multiLayerPerceptron 2 12 lightintensity Fa;ztorque linear;[motorlike,tanh] 500
fix 1 all multiLayerPerceptron 1 2 x;v Fa;ztorque [motorlike,tanh] 500
```

## Equations

_TODO_

# Supported values

## Activation functions

- relu
- sigmoid
- tanh
- linear
- motorlike (Gompertz)
- heaviside (actually from -1 to +1)

## Input and output
### 1 dimensional properties

- mass
- q
- radius
- rmass
- temperature
- heatflow
- qreward / _CUSTOM_
- Dr / _CUSTOM_
- Fa / _CUSTOM_
- zeta / _CUSTOM_
- lightintensity / _CUSTOM_
- clock / _CUSTOM_
- ang2D / _CUSTOM_
- ztorque / _CUSTOM_
- dreward / _CUSTOM_
- vfrac / _PERI_
- s0 / _PERI_
- eradius / _AWPMD_
- ervel / _AWPMD_
- erforce / _AWPMD_
- ervelforce / _AWPMD_
- conductivity / _RHEO_
- pressure / _RHEO_
- viscosity / _RHEO_
- rho / _SPH_
- drho / _SPH_
- esph / _SPH_
- desph / _SPH_
- cv / _SPH_
- dpdTheta / _DPD-REACT_
- edpd_temp / _DPD-MESO_
- area / _DIELECTRIC_
- ed / _DIELECTRIC_
- em / _DIELECTRIC_
- epsilon / _DIELECTRIC_
- curvature / _DIELECTRIC_
- q_unscaled / _DIELECTRIC_

### 2 dimensional properties

- cs / _AWPMD_
- csforce / _AWPMD_

### 3 dimensional properties

- x
- v
- f
- omega
- angmom
- torque
- x0 / _PERI_
- vforce / _AWPMD_
- vest / _SPH_

### 4 dimensional properties

- mu
- quat
