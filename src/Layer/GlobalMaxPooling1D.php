<?php
namespace Rindow\NeuralNetworks\Layer;

class GlobalMaxPooling1D extends AbstractGlobalMaxPooling
{
    protected int $rank = 1;
    protected string $defaultLayerName = 'globalmaxpooling1d';
}
