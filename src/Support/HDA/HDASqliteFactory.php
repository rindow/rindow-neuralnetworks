<?php
namespace Rindow\NeuralNetworks\Support\HDA;

class HDASqliteFactory implements HDAFactory
{
    public function open(string|object $filename, string $mode=null) : HDASqlite
    {
        return new HDASqlite($filename, $mode);
    }
}
