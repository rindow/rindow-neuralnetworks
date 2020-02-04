<?php
namespace Rindow\NeuralNetworks\Builder;

use Rindow\NeuralNetworks\Support\HDA\HDASqliteFactory;

class Utils
{
    public function HDA($type=null)
    {
        return new HDASqliteFactory();
    }
}
