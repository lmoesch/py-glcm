# Installation

    python setup.py install
    
# Usage
* [Grey Level Co-occurance Matrix](# Grey Level Co-occurance Matrix)
 * [glcm](# glcm.glcm)
 * xglcm
* GLCM Features

# Grey Level Co-occurance Matrix

## glcm.glcm

glcm.**glcm(**_array, dists, dirs, mode, symmetric=True, bins=256, normalized=True, check=True_**)**
  
Generates a GLCM.

<table>

<tr>
<td style="width: 120px; border:none; border-left: 4px solid #f0b37e;">
<h5>Parameters:</h5>
</td>
<td style="border:none;">
<b>array : </b> <i>array_like</i> <br />

&nbsp;&nbsp;Input image, either 2D or 3D.
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
<b>dists :</b> <i>array_like<i> <br />

&nbsp;&nbsp;Array of desired integer distances [d1, d2, ..., dn].
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
<b>dirs</b> : <i>array_like</i>

&nbsp;&nbsp;Array of desired integer directions [d1, d2, ..., dn]. 1 Corresponds to N, 2 to NE, 3 to E, ... and 8 to NW.

<blockquote><b>Caution!</b> Directions will be reordered ascendingly. To avoid confused outputs, please make sure to provide an ordered array.</blockquote>
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
<b>mode</b> : <i>string</i> <br />

Operation mode of the glcm. Features "sum" and "raw".

In raw-mode a glcm is generated for every combination of distances and directions.

In sum-mode all desired directions are added together so only one glcm per distance is generated. This is far more efficient than summing up afterwards.

</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
<b>symmetric</b> : <i>boolean, optional</i> <br />

<blockquote><b>Caution!</b> Running in symmetric mode will remove opposing directions and will replace directions 5 to 8 with their corresponding counterparts! To avoid confusing outputs please make sure only to use directions 1 to 4 in symmetric mode!
</blockquote>
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
<b>bins</b> : <i>int, optional</i> <br />

Number of bins. When input checking is enabled, the input image is binned to this number if the maximum image value exceeds the number of bins. This happens aswell if the input image is not of integer type. 

</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
<b>normalized</b> : <i>boolean, optional</i> <br />
Determines if the output shall be normalized or not.

<blockquote><b>Caution!</b> The normalization will fail if the input image has more than 10^15 pixels.</blockquote>
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
<b>check</b> : <i>boolean, optional</i> <br />
Determines if the input should be checked for correctness. 
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
<b>Returns:</b>
</td>
<td style="border:none;">
<b>glcm :</b> <i>ndarray</i> <br />

Array of glcm(s) with the following shape: 

[distances][directions][channels][bins][bins] 
</td>
</tr>
</table>

## glcm.xglcm

glcm.**xglcm(**_array, dists, dirs, mode, symmetric=True, bins=256, normalized=True, check=True_**)**
  
Generates a GLCM for every channel combination.

<table>

<tr>
<td style="width: 120px; border:none; border-left: 4px solid #f0b37e">
**Parameters:**
</td>
<td style="border:none;">
**array** : _array_like_

Input image with three dimensions with shape [dimx, dimy, channels]
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**dists** : _array_like_

Array of desired integer distances [d1, d2, ..., dn].
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**dirs** : _array_like_

Array of desired integer directions [d1, d2, ..., dn]. 1 Corresponds to N, 2 to NE, 3 to E, ... and 8 to NW.

**Caution!** Directions will be reordered ascendingly. To avoid confused outputs, please make sure to provide an ordered array.
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**mode** : _string_

Operation mode of the glcm. Features "sum" and "raw".

In raw-mode a glcm is generated for every combination of distances and directions.

In sum-mode all desired directions are added together so only one glcm per distance is generated. This is far more efficient than summing up afterwards.
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**symmetric** : _boolean, optional_

**Caution!** Running in symmetric mode will remove opposing directions and will replace directions 5 to 8 with their corresponding counterparts! To avoid confusing outputs please make sure only to use directions 1 to 4 in symmetric mode!
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**bins** : _int, optional_

Number of bins. When input checking is enabled, the input image is binned to this number if the maximum image value exceeds the number of bins. This happens aswell if the input image is not of integer type. 

</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**normalized** : _boolean, optional_
Determines if the output shall be normalized or not.

**Caution!** The normalization will fail if the input image has more than 10^15 pixels.
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**check** : _boolean, optional_
Determines if the input should be checked for correctness. 
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
**Returns:**
</td>
<td style="border:none;">
**glcm** : _ndarray_

Array of glcm(s) with the following shape: 

[distances][directions][source channels][target channels][bins][bins] 
</td>
</tr>
<table>

# GLCM Features  

## glcm.glcm_features

glcm.**glcm_features(**_array, features, symmetric=True, normalized=True_**)**
  
Calculates features from a given set of GLCMs

<table>

<tr>
<td style="width: 120px; border:none; border-left: 4px solid #f0b37e">
**Parameters:**
</td>
<td style="border:none;">
**array** : _array_like_

Array of glcms with shape where the last two axes covering the glcm entries.
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**features** : _int_

Binary input that determines the desired features. See below.

</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**symmetric** : _boolean, optional_

Indicates if the input GLCM(s) are symmetric or not.
</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
</td>
<td style="border:none;">
**normalized** : _boolean, optional_
Determines if the input is normalized.

</td>
</tr>
<tr>
<td style="border:none;border-left: 4px solid #f0b37e">
**Returns:**
</td>
<td style="border:none;">
**glcm** : _dict_

Dictionary of features.

</td>
</tr>
<table>

### Supported Features

#### Angular Second Moment
glcm.**asm**, dictionary: _"ASM"_

<img src="/doc/latex/img/asm.png" width="200">

Measurement of homogeneous patterns in the image.

#### Contrast
glcm.**contrast**, dictionary: _"Contrast"_

#### Correlation
glcm.**correl**, dictionary: _"Correlation"_

Not implemented yet

#### Autocorrelation
glcm.**autocorrel**, dictionary: _"Auto Correlation"_

#### Sum of Squares
glcm.**ssq**, dictionary: _"SSQ"_

Not implemented yet

#### Inverse Difference Moment
glcm.**idm**, dictionary: _"IDM"_

#### Inverse Difference
glcm.**idf**, dictionary: _"IDF"_

#### Sum Average
glcm.**sumavg**, dictionary: _"Sum Average"_

#### Sum Variance
glcm.**sumvar**, dictionary: _"Sum Variance"_

#### Sum Entropy
glcm.**sumentrp**, dictionary: _"Sum Entropy"_

#### Difference Average
glcm.**diffavg**, dictionary: _"Diff Average"_

#### Difference Variance
glcm.**diffvar**, dictionary: _"Diff Variance"_

Not implemented yet

#### Difference Entropy
glcm.**diffentrp**, dictionary: _"Diff Entropy"_

#### Cluster Prominence
glcm.**clusterprom**, dictionary: _"Cluster Prominence"_

#### Cluster Shade
glcm.**clustershade**, dictionary: _"Cluster Shade"_

#### Cluster Tendency
glcm.**clustertend**, dictionary: _"Cluster Tendency"_

#### Dissimilarity
glcm.**Dissim**, dictionary: _"Dissimilarity"_




  